import os
import random
from enum import Enum, unique
from typing import Dict, List
import json

import matplotlib
import matplotlib.pylab as plt
import seaborn as sns

sns.set_style()
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams["font.family"] = "Rubik"
sns.set_context("paper")

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from pit.dynamics.kinematic_bicycle import Bicycle
from pit.dynamics.unicycle import Unicycle
from pit.integration import RK4, Euler
from tensorboardX import SummaryWriter
from torch.multiprocessing import Pool, Process, freeze_support
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm
from vmp.data_utils import (RACE_TEST_LIST, TEST_LIST, TRAIN_LIST, VAL_LIST,
                            TraceRelativeDataset, get_map_points, map_x, map_y)
from vmp.losses import (average_displacement_error, custom_loss_func,
                        final_displacement_error, heading_error,
                        standard_loss_func, iou_metric)
from vmp.networks import LSTMPredictor, LSTMPredictorPCMP, get_model
from vmp.utils import Curvature, DynamicModel, Method, set_seed
from vmp.viz_utils import create_debug_plot

freeze_support()

HEADLESS = True

train_frame = pd.read_pickle("../../data/train_data.pkl")
test_frame = pd.read_pickle("../../data/test_data.pkl")
val_frame = pd.read_pickle("../../data/val_data.pkl")
full_frame = pd.read_pickle("../../data/final_data.pkl")

no_race_train_frame = train_frame[
    train_frame["selected_lane"].apply(
        lambda x: True if x in ["left", "center", "right"] else False
    )
]
no_race_val_frame = val_frame[
    val_frame["selected_lane"].apply(
        lambda x: True if x in ["left", "center", "right"] else False
    )
]
no_race_test_frame = test_frame[
    test_frame["selected_lane"].apply(
        lambda x: True if x in ["left", "center", "right"] else False
    )
]
race_test_frame = full_frame[
    full_frame["selected_lane"].apply(lambda x: True if x in ["race"] else False)
]
train_dataset = TraceRelativeDataset(train_frame, curve=True)
test_dataset = TraceRelativeDataset(test_frame, curve=True)
val_dataset = TraceRelativeDataset(val_frame, curve=True)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=8)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=8)


def test_model(
    config_dict,
    net,
    dataloader,
    loss_func,
    epoch=None,
    curriculum_steps=None,
    dataset_name=None,
    histogram_results=False,
    iou=False,
    image=True,
    seed=None,
):
    DEVICE = config_dict["DEVICE"]
    step_size = 60 // config_dict["control_outputs"]
    cum_loss = 0.0
    net.eval()
    fde, ade = 0.0, 0.0
    cum_head_err = 0.0
    iou_err = 0.0
    element_results = {
        "fde": [],
        "ade": [],
        "heading_error": [],
        "loss": [],
    }

    if seed is not None:
        set_seed(seed)
    
    with torch.no_grad():
        for input_data, last_pose, target_data in dataloader:
            net.zero_grad()
            input_data = input_data.to(DEVICE)
            last_pose = last_pose.to(DEVICE)
            if config_dict["method"] is Method.PIMP:
                outp, control_outputs = net.predict(input_data, last_pose)
                target_data = target_data.to(DEVICE)
                if epoch >= curriculum_steps or not config_dict["curriculum"]:
                    loss = loss_func(outp, target_data, control_outputs)
                    end_index = config_dict["horizon"]
                else:
                    current_step = epoch // config_dict["eps_per_input"]
                    end_index = min(
                        step_size * (current_step + 1), config_dict["horizon"]
                    )
                    loss = loss_func(
                        outp[:, :end_index],
                        target_data[:, :end_index],
                        control_outputs[:, :end_index],
                    )
            elif config_dict["method"] is Method.LSTM:
                outp = net.predict(input_data, last_pose)
                target_data = target_data.to(DEVICE)
                if epoch >= curriculum_steps or not config_dict["curriculum"]:
                    loss = loss_func(outp, target_data)
                    end_index = config_dict["horizon"]
                else:
                    current_step = epoch // config_dict["eps_per_input"]
                    end_index = min(
                        step_size * (current_step + 1), config_dict["horizon"]
                    )
                    loss = loss_func(outp[:, :end_index], target_data[:, :end_index])
            elif config_dict["method"] in [Method.CTRV, Method.CTRA]:
                outp = net.predict(input_data, last_pose)
                target_data = target_data.to(DEVICE)
                loss = loss_func(outp, target_data)
                end_index = config_dict["horizon"]
            fde += final_displacement_error(outp, target_data).cpu().numpy()
            ade += average_displacement_error(outp, target_data).cpu().numpy()
            if iou:
                iou_err += iou_metric(outp, target_data).cpu().numpy()
            cum_head_err += heading_error(outp, target_data).cpu().numpy()
            cum_loss += loss.item()
            if histogram_results:
                for trace in range(input_data.shape[0]):
                    element_results["ade"].append(
                        average_displacement_error(outp[trace], target_data[trace])
                        .cpu()
                        .numpy()
                    )
                    element_results["fde"].append(
                        final_displacement_error(outp[trace], target_data[trace])
                        .cpu()
                        .numpy()
                    )
                    element_results["heading_error"].append(
                        heading_error(outp[trace], target_data[trace]).cpu().numpy()
                    )
                    if config_dict["method"] is Method.PIMP:
                        element_results["loss"].append(
                            loss_func(
                                outp[trace],
                                target_data[trace],
                                control_outputs[trace],
                            ).item()
                        )
                    else:
                        element_results["loss"].append(
                            loss_func(outp[trace], target_data[trace]).item()
                        )
                    

    if image:
        fig, ax = create_debug_plot(
            net,
            train_dataset=train_dataloader.dataset,
            test_dataset=test_dataloader.dataset,
            val_dataset=val_dataloader.dataset,
            curvature=config_dict["curvature"],
            DEVICE=DEVICE,
            full_frame=full_frame,
            dataset_name=dataset_name,
        )
    else:
        fig=None
    cum_loss /= len(dataloader.dataset)  # Normalize for data items
    cum_loss *= end_index / config_dict["horizon"]  # Normalize for curriculum length
    ade = ade / len(dataloader.dataset)
    fde = fde / len(dataloader.dataset)
    cum_head_err = cum_head_err / len(dataloader.dataset)
    iou_err = iou_err/len(dataloader.dataset)

    return ade, fde, cum_head_err, iou_err, cum_loss, fig, end_index, element_results


def train_loop(
    config_dict: Dict,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    val_dataloader: DataLoader,
    aux_test_dataloader: DataLoader,
):
    DEVICE = config_dict["DEVICE"]
    net, opt, directory, config_dict = get_model(config_dict)
    net.to(DEVICE)
    writer = SummaryWriter(directory)

    train_losses = list()
    test_losses = list()
    val_losses = list()
    race_losses = list()
    train_ades, val_ades, test_ades, race_ades = list(), list(), list(), list()
    train_fdes, val_fdes, test_fdes, race_fdes = list(), list(), list(), list()
    train_ious, val_ious, test_ious, race_ious = list(), list(), list(), list()
    (
        train_heading_errors,
        val_heading_errors,
        test_heading_errors,
        race_heading_errors,
    ) = (list(), list(), list(), list())

    step_size = 60 // config_dict["control_outputs"]
    curriculum_steps = config_dict["eps_per_input"] * config_dict["control_outputs"]
    loss_func = (
        standard_loss_func if config_dict["loss_func"] == "disp" else custom_loss_func
    )

    with tqdm(total=config_dict["epochs"], unit="epochs", disable=HEADLESS) as progbar:
        for epoch in range(config_dict["epochs"]):
            progbar.set_description("TRAINING")
            if config_dict["method"] not in [Method.CTRV, Method.CTRA]:
                cum_train_loss = 0.0
                net.train()
                fde, ade = 0.0, 0.0
                cum_head_err = 0.0
                for input_data, last_pose, target_data in train_dataloader:
                    opt.zero_grad()
                    input_data = input_data.to(DEVICE)
                    last_pose = last_pose.to(DEVICE)
                    if config_dict["method"] is Method.PIMP:
                        outp, control_outputs = net.predict(input_data, last_pose)
                        target_data = target_data.to(DEVICE)
                        if epoch >= curriculum_steps or not config_dict["curriculum"]:
                            loss = loss_func(outp, target_data, control_outputs)
                            end_index = config_dict["horizon"]
                        else:
                            current_step = epoch // config_dict["eps_per_input"]
                            end_index = min(
                                step_size * (current_step + 1), config_dict["horizon"]
                            )
                            loss = loss_func(
                                outp[:, :end_index],
                                target_data[:, :end_index],
                                control_outputs[:, :end_index],
                            )
                    elif config_dict["method"] is Method.LSTM:
                        outp = net.predict(input_data, last_pose)
                        target_data = target_data.to(DEVICE)
                        if epoch >= curriculum_steps or not config_dict["curriculum"]:
                            loss = loss_func(outp, target_data)
                            end_index = config_dict["horizon"]
                        else:
                            current_step = epoch // config_dict["eps_per_input"]
                            end_index = min(
                                step_size * (current_step + 1), config_dict["horizon"]
                            )
                            loss = loss_func(
                                outp[:, :end_index], target_data[:, :end_index]
                            )
                    loss.backward()
                    opt.step()
                    with torch.no_grad():
                        fde += final_displacement_error(outp, target_data).cpu().numpy()
                        ade += average_displacement_error(outp, target_data).cpu().numpy()
                        cum_head_err += heading_error(outp, target_data).cpu().numpy()
                    cum_train_loss += loss.item()
                if config_dict["curriculum"] and epoch <= curriculum_steps and not HEADLESS:
                    print(
                        f"Epoch {epoch}, Current Step {current_step}, End Index {end_index}, Step Size {step_size}"
                    )
                if config_dict["curriculum"]:
                    writer.add_scalar("end_index", end_index, epoch)
                # writer.add_scalar("lr", lr_scheduler.get_last_lr(), epoch)
                # lr_scheduler.step()
                train_fig, train_ax = create_debug_plot(
                    net,
                    train_dataset=train_dataloader.dataset,
                    test_dataset=test_dataloader.dataset,
                    val_dataset=val_dataloader.dataset,
                    curvature=config_dict["curvature"],
                    DEVICE=DEVICE,
                    full_frame=full_frame,
                    dataset_name="train",
                )
                cum_train_loss /= len(train_dataloader.dataset)  # Normalize for data items
                cum_train_loss *= (
                    end_index / config_dict["horizon"]
                )  # Normalize for curriculum length
                ade = ade / len(train_dataloader.dataset)
                fde = fde / len(train_dataloader.dataset)
                cum_head_err = cum_head_err / len(train_dataloader.dataset)
            
            else:
                # CTRV or CTRA
                ade, fde, cum_head_err, iou_err, cum_train_loss, train_fig, end_index, _ = test_model(
                    config_dict=config_dict,
                    net=net,
                    dataloader=train_dataloader,
                    loss_func=loss_func,
                    epoch=epoch,
                    curriculum_steps=curriculum_steps,
                    dataset_name="train",
                    histogram_results=False,
                    iou=True,
                )
                train_ious.append(iou_err)
                writer.add_scalar("train/IoU", ade, epoch)

            train_ades.append(ade)
            train_fdes.append(fde)
            train_heading_errors.append(cum_head_err)
            train_losses.append(cum_train_loss)
            

            writer.add_scalar("train/ADE", ade, epoch)
            writer.add_scalar("train/FDE", fde, epoch)
            writer.add_scalar("train/heading_error", cum_head_err, epoch)
            writer.add_figure("train/example_fig", train_fig, epoch)
            writer.add_scalar("train/loss", cum_train_loss, epoch)
            writer.flush()
            plt.close("all")

            progbar.set_description("VALIDATION")
            ade, fde, cum_head_err, iou_err, cum_val_loss, val_fig, end_index, _ = test_model(
                config_dict=config_dict,
                net=net,
                dataloader=val_dataloader,
                loss_func=loss_func,
                epoch=epoch,
                curriculum_steps=curriculum_steps,
                dataset_name="val",
                histogram_results=False,
            )

            val_ades.append(ade)
            val_fdes.append(fde)
            val_heading_errors.append(cum_head_err)
            val_losses.append(cum_val_loss)
            #val_ious.append(iou_err)

            writer.add_scalar("val/ADE", ade, epoch)
            writer.add_scalar("val/FDE", fde, epoch)
            writer.add_scalar("val/IoU", iou_err, epoch)
            writer.add_scalar("val/heading_error", cum_head_err, epoch)
            writer.add_figure("val/example_fig", val_fig, epoch)
            writer.add_scalar("val/loss", cum_val_loss, epoch)
            writer.flush()
            plt.close("all")

            if (epoch) % 10 == 0:
                ade, fde, cum_head_err, iou_err, cum_test_loss, test_fig, end_index, element_results = test_model(
                    config_dict=config_dict,
                    net=net,
                    dataloader=test_dataloader,
                    loss_func=loss_func,
                    epoch=epoch,
                    curriculum_steps=curriculum_steps,
                    dataset_name="test",
                    histogram_results=True,
                    iou=True
                )

                test_ades.append(ade)
                test_fdes.append(fde)
                test_ious.append(iou_err)
                test_heading_errors.append(cum_head_err)
                test_losses.append(cum_test_loss)

                writer.add_scalar("test/ADE", ade, epoch)
                writer.add_scalar("test/FDE", fde, epoch)
                writer.add_scalar("test/IoU", iou_err, epoch)
                writer.add_scalar("test/heading_error", cum_head_err, epoch)
                writer.add_figure("test/example_fig", test_fig, epoch)
                writer.add_scalar("test/loss", cum_test_loss, epoch)
                writer.add_histogram(
                    "test/ADE/distributions", np.array(element_results["ade"]), epoch
                )
                writer.add_histogram(
                    "test/FDE/distributions", np.array(element_results["fde"]), epoch
                )
                writer.add_histogram(
                    "test/heading_error/distributions", np.array(element_results["heading_error"]), epoch
                )
                writer.add_histogram(
                    "test/loss/distributions", np.array(element_results["loss"]), epoch
                )

                if aux_test_dataloader is not None:
                    ade, fde, cum_head_err, iou_err, cum_race_test_loss, race_fig, end_index, element_results = test_model(
                        config_dict=config_dict,
                        net=net,
                        dataloader=aux_test_dataloader,
                        loss_func=loss_func,
                        epoch=epoch,
                        curriculum_steps=curriculum_steps,
                        dataset_name="race_test",
                        histogram_results=True,
                        iou=True
                    )

                    race_ades.append(ade)
                    race_fdes.append(fde)
                    race_ious.append(iou_err)
                    race_heading_errors.append(cum_head_err)
                    race_losses.append(cum_race_test_loss)

                    writer.add_scalar("race_test/ADE", ade, epoch)
                    writer.add_scalar("race_test/FDE", fde, epoch)
                    writer.add_scalar("race_test/IoU", iou_err, epoch)
                    writer.add_scalar("race_test/head_err", cum_head_err, epoch)
                    writer.add_figure("race_test/example_fig", race_fig, epoch)
                    writer.add_scalar("race/loss", cum_race_test_loss, epoch)
                    writer.add_histogram(
                        "race_test/ADE/distributions", np.array(element_results["ade"]), epoch
                    )
                    writer.add_histogram(
                        "race_test/FDE/distributions", np.array(element_results["fde"]), epoch
                    )
                    writer.add_histogram(
                        "race_test/heading_error/distributions", np.array(element_results["heading_error"]), epoch
                    )
                    writer.add_histogram(
                        "race_test/loss/distributions", np.array(element_results["loss"]), epoch
                    )
                    plt.close("all")
                    writer.flush()

            if cum_val_loss <= min(val_losses):
                print("New best model found!")
                # Delete old best model
                try:
                    os.remove(f"{directory}/best_model.pt")
                    os.remove(f"{directory}/best_model_meta.pt")
                except FileNotFoundError:
                    pass # No old best model found
                # Save new best model
                torch.save(net.state_dict(), f"{directory}/best_model.pt")
                config_copy = config_dict.copy()
                config_copy.update(output_queue=None)
                torch.save(
                    {"epoch": epoch, "config": config_copy},
                    f"{directory}/best_model_meta.pt",
                )
            if config_dict["save_all"]:
                torch.save(net.state_dict(), f"{directory}/{epoch}_model.pt")
                config_copy = config_dict.copy()
                config_copy.update(output_queue=None)
                torch.save(
                    {"epoch": epoch, "config": config_copy},
                    f"{directory}/{epoch}_meta.pt",
                )
            if not HEADLESS:
                progbar.write(
                    f"Epoch {epoch} | Train Loss: {cum_train_loss} | Test Loss: {cum_test_loss}"
                )
            progbar.update()
            if "output_queue" in config_dict:
                # We have an output Queue
                config_dict["output_queue"].put({
                    "message_type": "progress",
                    "content": 1,
                })

    # Write the full scalar content to file
    #     {
    #         "train_losses": train_losses,
    #         "test_losses": test_losses,
    #         "val_losses": val_losses,
    #         "race_losses": race_losses,
    #         "train_ades": train_ades,
    #         "test_ades": test_ades,
    #         "val_ades": val_ades,
    #         "race_ades": race_ades,
    #         "train_fdes": train_fdes,
    #         "test_fdes": test_fdes,
    #         "val_fdes": val_fdes,
    #         "race_fdes": race_fdes,
    #         "train_heading_errors": train_heading_errors,
    #         "test_heading_errors": test_heading_errors,
    #         "val_heading_errors": val_heading_errors,
    #         "race_heading_errors": race_heading_errors,
    #     }
    # )
    # metadata.to_pickle(f"{directory}/training_metadata.pkl")

    torch.save(net.state_dict(), f"{directory}/last_model.pt")
    config_copy = config_dict.copy()
    config_copy.update(output_queue=None)
    torch.save(
        {
            "epoch": epoch,
            "config": config_copy,
            "train_losses": train_losses,
            "test_losses": test_losses,
            "val_losses": val_losses,
            "race_losses": race_losses,
            "train_ades": train_ades,
            "test_ades": test_ades,
            "val_ades": val_ades,
            "race_ades": race_ades,
            "train_fdes": train_fdes,
            "test_fdes": test_fdes,
            "val_fdes": val_fdes,
            "race_fdes": race_fdes,
            "train_heading_errors": train_heading_errors,
            "test_heading_errors": test_heading_errors,
            "val_heading_errors": val_heading_errors,
            "race_heading_errors": race_heading_errors,
            "train_ious": train_ious,
            "val_ious": val_ious,
            "test_ious": test_ious,
            "race_ious": race_ious,
        },
        f"{directory}/last_model_meta.pt",
    )


    writer.close()
    del net, opt

    return (
        config_copy,
        {
            "training_loss": cum_train_loss,
            "test_loss": cum_test_loss,
            "ade": ade,
            "fde": fde,
            "iou": test_ious[-1],
        },
    )


def train(config_dict: dict):
    print(config_dict)
    base_args = dict(
        hidden_dim=32,
        epochs=100,
        control_outputs=10,
        curriculum=False,
        eps_per_input=2,
        prefix="runs/final-toy/other-experiments/",
        batch_size=512,
        DEVICE="cpu",
        wheelbase=0.3302,
        residual=False,
        model=DynamicModel.BICYCLE,
        robustness=False,
        wd=1e-5,
        lr=1e-3,
        momentum=0.8,
        horizon=60,
        timestep=0.01,
        loss_func="custom",
        save_all=False,
    )
    base_args.update(config_dict)
    config_dict = base_args  # Old switcheroo
    assert "method" in base_args.keys()
    assert "curvature" in base_args.keys()

    set_seed()

    train_dataloader = None
    test_dataloader = None
    val_dataloader = None
    race_dataloader = None

    if not config_dict["robustness"]:
        train_dataset = TraceRelativeDataset(
            train_frame,
            curve=config_dict["curvature"] is Curvature.CURVATURE,
            random_noise=True
        )
        test_dataset = TraceRelativeDataset(
            test_frame,
            curve=config_dict["curvature"] is Curvature.CURVATURE,
            random_noise=True
        )
        val_dataset = TraceRelativeDataset(
            val_frame,
            curve=config_dict["curvature"] is Curvature.CURVATURE,
            random_noise=True
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=config_dict["batch_size"], shuffle=True
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=config_dict["batch_size"], shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=config_dict["batch_size"], shuffle=True
        )
    elif config_dict["robustness"]:
        train_dataset = TraceRelativeDataset(
            no_race_train_frame,
            curve=config_dict["curvature"] is Curvature.CURVATURE,
            random_noise=True
        )
        val_dataset = TraceRelativeDataset(
            no_race_val_frame,
            curve=config_dict["curvature"] is Curvature.CURVATURE,
            random_noise=True
        )
        test_dataset = TraceRelativeDataset(
            no_race_test_frame,
            curve=config_dict["curvature"] is Curvature.CURVATURE,
            random_noise=True
        )
        race_dataset = TraceRelativeDataset(
            race_test_frame, 
            curve=config_dict["curvature"] is Curvature.CURVATURE,
            random_noise=True
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=config_dict["batch_size"], shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=config_dict["batch_size"], shuffle=True
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=config_dict["batch_size"], shuffle=True
        )
        race_dataloader = DataLoader(
            race_dataset, batch_size=config_dict["batch_size"], shuffle=True
        )
        

    params, results = train_loop(
        config_dict,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        val_dataloader=val_dataloader,
        aux_test_dataloader=race_dataloader,
    )
    return params, results


class GPURunner:
    """Generates runners for jobs, which listen to a queue for jobs"""

    def __init__(self, gpu_id: str):
        self.gpu_id = gpu_id

    def model_training_process(
        self, input_queue: multiprocessing.Queue, output_queue: multiprocessing.Queue
    ):
        DEVICE = self.gpu_id
        while not input_queue.empty():
            job_definition = input_queue.get()
            job_definition["DEVICE"] = self.gpu_id
            job_definition["output_queue"] = output_queue
            results = train(job_definition)
            output_queue.put({
                "message_type": "result",
                "parameters": results[0],
                "results": results[1],
            })
        #output_queue.close()


def main(prefix, jobs, gpus, procs_per_gpu, robustness, runner_specs=None):
    total_epochs = np.sum([job["epochs"] for job in jobs])
    for i in range(len(jobs)):
        jobs[i]["job_id"] = i
    multiprocessing.set_start_method("spawn")
    jobs_queue = multiprocessing.Manager().Queue()
    results_queue = multiprocessing.Manager().Queue()

    if runner_specs is None:
        gpus *= procs_per_gpu
    else:
        gpus = runner_specs

    objs_list = [GPURunner(gpu) for gpu in gpus]

    proclist = [
        multiprocessing.Process(
            target=runner.model_training_process, args=(jobs_queue, results_queue)
        )
        for runner in objs_list
    ]

    for proc in proclist:
        proc.start()

    for j in jobs:
        jobs_queue.put(j)

    print(jobs_queue.qsize())


    procs_alive = True
    hyp_params_writer = SummaryWriter(prefix)
    hyperparameters = []
    done = 0

    with tqdm(total=total_epochs, unit="Total Epochs") as progbar:
        while procs_alive:
            try:
                message = results_queue.get(timeout=20)
                if message["message_type"] == "progress":
                    progbar.update(message["content"])
                elif message["message_type"] == "result":
                    hyp_params_writer.add_hparams(message["parameters"], message["results"])
                    params = message["parameters"]
                    params.update(message["results"])
                    hyperparameters.append(params)
                    progbar.write(
                        "Got a result: Params:{} \n Accuracy {}".format(message["parameters"], message["results"])
                    )
                    torch.save(hyperparameters, f"{prefix}/hyperparameters.pkl")
            except:
                pass
            any_alive = False
            for proc in proclist:
                if proc.is_alive():
                    any_alive = True
            procs_alive = any_alive

        for proc in proclist:
            proc.join()
    
    # Write hyperparameters to a file
    torch.save(hyperparameters, f"{prefix}/hyperparameters.pkl")

    
if __name__ == "__main__":
    freeze_support()

    prefix = "runs/final-toy/vel-test/"

    jobs = [
        dict(
            method=Method.PIMP,
            curvature=Curvature.CURVATURE,
            robustness=False,
            hidden_dim=16,
            epochs=100,
            control_outputs=10,
            curriculum=True,
            eps_per_input=1,
            prefix=prefix,
        ),
        dict(
            method=Method.PIMP,
            curvature=Curvature.CURVATURE,
            robustness=False,
            hidden_dim=4,
            epochs=100,
            control_outputs=60,
            curriculum=True,
            eps_per_input=3,
            prefix=prefix,
        ),
        dict(
            method=Method.PIMP,
            curvature=Curvature.CURVATURE,
            robustness=False,
            hidden_dim=4,
            epochs=100,
            control_outputs=60,
            curriculum=False,
            prefix=prefix,
        ),
        dict(
            method=Method.LSTM,
            curvature=Curvature.CURVATURE,
            robustness=False,
            hidden_dim=4,
            epochs=100,
            prefix=prefix,
        ),
    ]

    gpus = [f"cuda:{i}" for i in range(3)]
    procs_per_gpu = 4

    train(jobs[-1])

    # main(prefix, jobs, gpus, procs_per_gpu, robustness)
