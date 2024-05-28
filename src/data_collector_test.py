import os
import json
import numpy as np
import torch
from PIL import Image
import imageio
import matplotlib.pyplot as plt
import time

class DataLoggerTest:

    def __init__(self, experiment_config,
                H, W,
                experiment_name = 'experiment'):
        print("DataLogger Test init")
        print(f"Experiment: {experiment_name}")
        config = experiment_config['data_collector']
        self.saved_property_name = config['save_property']
        for key in experiment_config:
            if self.saved_property_name in experiment_config[key]:
                self.saved_property_val = experiment_config[key][self.saved_property_name]

        self.history = experiment_config['training'].get("save_history", False)

        iterations = experiment_config['training']['iterations']
        self.gif_interval = iterations // config['num_of_gif_frames']
        self.xyz_interval = iterations // config['num_of_xyz_frames']
        self.viewer_interval = iterations // config['num_of_viewer_frames']

        self.experiment_name = experiment_name 
        self.H = H
        self.W = W

        self.log_all = config['log_all']

        self.loss_path = config['loss_path']
        self.img_path = config['img_path']
        self.means_path = config['means_path']
        
        self.losses = []
        self.images = []
        self.means = []
        self.points_viewer = []
        self.bg_penalties = []
        self.cameras = {}
        self.points_3D = None
        self.grads = {}
        

        self.out_dir = os.path.join(os.getcwd(), "data_log", experiment_name)
        self.training_dir = os.path.join(self.out_dir, "training")
        self.loss_dir = os.path.join(self.out_dir, "losses")
        self.points_dir = os.path.join(self.out_dir, "points")
        
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.loss_dir, exist_ok=True)
        os.makedirs(self.training_dir, exist_ok=True)
        os.makedirs(self.points_dir, exist_ok=True)

        self.save_config(experiment_config)

    def save_config(self, config):
        with open(f"{self.out_dir}/config.json", "w") as f:
            json.dump(config, f)
        f.close()
        
    def log_iter_data(self,
                 iter,
                 loss,
                 img,
                 camera_idx,
                 c2w,
                 means,
                 points2d,
                 rgbs,
                 opacities):
        camera_idx = camera_idx.item()
        c2w = c2w.cpu().numpy()
        concat_means_rgbs = self.get_xys_rgbs(points2d, rgbs)
        concat_points_rgb_opa = self.get_means_rgb_opacities(means, rgbs, opacities)
        self.log_loss(loss)
        self.log_camera(camera_idx, c2w, loss, img)
        self.log_xys(iter, concat_means_rgbs)
        self.log_points(iter, concat_points_rgb_opa)
        self.log_image(iter, img)
    
    def log_post_iter_data(self, means, rgbs, opacities):
        points_3D_with_color = self.get_means_rgb_opacities(means, rgbs, opacities)
        self.log_3D_points(points_3D_with_color)
                
    def get_xys_rgbs(self, xys, rgbs):
        return torch.cat([xys.detach(), rgbs.detach()], dim=1).cpu().numpy()
    
    def get_means_rgb_opacities(self, means, rgbs, opacities):
        return torch.cat([means.detach(), rgbs.detach(), opacities.detach()], dim=1).cpu().numpy()
 
    def log_loss(self, loss):
        self.losses.append(loss)
            
    def log_image(self, iter, img):
        if iter % self.gif_interval == 0:
            img = (img.cpu().numpy() * 255).astype(np.uint8)
            self.images.append(img)
    
    def log_xys(self, iter, means):
        if iter % self.xyz_interval == 0:
            self.means.append(means)

    def log_3D_points(self,points):
        self.points_3D = points
    
    def log_points(self, iter, points):
        if iter % self.viewer_interval == 0:
            self.points_viewer.append(points)

    def log_camera(self, camera_idx, c2w, loss, img):
        if camera_idx not in self.cameras:
            img = (img.cpu().numpy() * 255).astype(np.uint8)
            self.cameras[camera_idx] = {"c2w": c2w, 
                                        "images": [img],
                                        "losses": [loss]}
        else:
            img = (img.cpu().numpy() * 255).astype(np.uint8)
            self.cameras[camera_idx]["images"].append(img) 
            self.cameras[camera_idx]["losses"].append(loss)
            
    def log_bg_penalty(self,bg_penalty):
        self.bg_penalties.append(bg_penalty)
     
    def log_mlp_grads(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_means = param.grad.norm().item()
                if name not in self.grads:
                    self.grads[name] = [grad_means]
                else:
                    self.grads[name].append(grad_means)
            
    def visualize_grads(self):
        
        for name, grads in self.grads.items():
            plt.plot(range(1, len(grads) + 1), grads)
            plt.xlabel("Iterations")
            plt.ylabel("Gradient Norm")
            plt.title(f"Gradient Norm over iterations\n{name}")
            #save the plot
            plt.savefig(f"{self.loss_dir}/{name}_grads.png")
            plt.close()

    def make_final_image(self):
       final_image = Image.fromarray(self.images[-1])
       final_image.save(f"{self.training_dir}/final_image.png")
    
    def create_gif(self, images, out_dir, name):

        images[0].save(
            os.path.join(out_dir, name),
            save_all=True,
            append_images=images[1:],
            optimize=False,
            duration=100,
            loop=0
        )

    def time_it(self, title, timed_function, *args):
        start = time.time()
        timed_function(*args)
        end = time.time()
        print(f"Time {title}: {end - start}")

    def create_points_visualization(self, xys_interval):
        
        for idx, points in enumerate(self.means):
            idx = idx + 1

            #color the points according to the rgb values
            plt.scatter(points[:, 0], points[:, 1], c=points[:, 2:5])
            plt.gca().add_patch(plt.Rectangle((0, 0), 
                                              self.W, self.H, 
                                              linewidth=2, 
                                              edgecolor='r', 
                                              facecolor='none'))

            plt.savefig(f"{self.points_dir}/point_distribution_{idx * xys_interval}.png")
            plt.close()

        images = [
            Image.open(f"{self.points_dir}/point_distribution_{(i + 1) * xys_interval}.png") 
            for i in range(len(self.means))
            ]
        
        self.create_gif(images, self.points_dir, "point_distribution2.gif")

    def fast_create_points_visualization(self, xys_interval):

        writer = imageio.get_writer(f"{self.points_dir}/point_distribution.gif", duration=0.1)
        fig, ax = plt.subplots()
        scatter = ax.scatter([], [])
        rectangle = plt.Rectangle((0, 0), self.W, self.H, linewidth=2, edgecolor='r', facecolor='none')
        
        for idx, points in enumerate(self.means):
            #points = points[::10]
            scatter.set_offsets(points[:, :2])
            scatter.set_color(points[:, 2:5])
    
            ax.add_patch(rectangle)
            ax.set_title(self.experiment_name)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(image)
            plt.close(fig)
        writer.close()


    def create_loss_visualization(self):
        
        plt.plot(range(1, len(self.losses) + 1), self.losses, label = "Loss")
        plt.plot(range(1, len(self.bg_penalties) + 1), self.bg_penalties, label = "Bg penalty")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        property_text = f"{self.saved_property_name}: {self.saved_property_val}"
        plt.title(f"Loss over iterations\n{self.experiment_name}\n{property_text}")
        #put the final loss value on the middle of the plot
        plt.text(len(self.losses) // 2, self.losses[-1], f"{self.losses[-1]:.2f}", ha='center', va='bottom')
        plt.savefig(f"{self.loss_dir}/loss.png")
        plt.legend()
        plt.close()

    def create_training_visualization(self):
        images = [Image.fromarray(frame) for frame in self.images]
        self.create_gif(images, self.training_dir, "training.gif")

    def create_3D_visualization(self):
        points = self.points_3D[::100]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], 
                   points[:, 1],
                   points[:, 2], 
                   c=points[:, 3:6],
                   alpha=points[:, -1])
        
        for _, camera_data in self.cameras.items():
            c2w = camera_data['c2w'] / 4
            camera_position = c2w[:, 3]
            for i, color in enumerate(['r', 'g', 'b']):
                if color == "b":
                    axis = - c2w[:3, i] * 2
                else:
                    axis = c2w[:3, i]
                ax.quiver(*camera_position, *axis, color=color)
        
        plt.title(self.experiment_name)
        plt.savefig(f"{self.points_dir}/3D_point_distribution.png")
        plt.close()
        
    def save_results(self):
        avg_loss = sum(self.losses) / len(self.losses)
        loss_variance = np.var(self.losses)
        loss_std = np.std(self.losses)
        final_loss = self.losses[-1]
        
        results = {
            "avg_loss": avg_loss,
            "loss_variance": loss_variance,
            "loss_std": loss_std,
            "final_loss": final_loss
        }
        with open(f"{self.out_dir}/results.json", "w") as f:
            json.dump(results, f)
    
    def save_history(self):
        if self.history:
            np.save(f"{self.points_dir}/points.npy", self.points_viewer)
    
    def plot_losses_for_cameras(self, ax, cameras, color, label_suffix = None):
        for camera_idx, camera_data in cameras:
            if label_suffix is not None:
                label = f"Camera {camera_idx} ({label_suffix})"
            else:
                label = None
            ax.plot(range(1, len(camera_data['losses']) + 1),
                     camera_data['losses'],
                     color = color,
                     label = label)     
    def plot_images_for_cameras(self, fig, start_idx, cameras, label_suffix):
        for i, (camera_idx, camera_data) in enumerate(cameras):
            ax = fig.add_subplot(4, 3, i + start_idx)
            ax.imshow(camera_data['images'][-1])
            ax.axis('off')
            ax.set_title(f"Camera {camera_idx} ({label_suffix})")
                    
    def save_loss_for_cameras(self):
        
        sorted_cameras = sorted(self.cameras.items(), key = lambda x: x[1]['losses'][-1])
        mid_index = len(sorted_cameras) // 2
        
        best_cameras = sorted_cameras[:3]
        worst_cameras = sorted_cameras[-3:]
        rest_of_cameras = sorted_cameras[3:-3]
        middle_cameras = sorted_cameras[mid_index-1:mid_index+2]
        
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(4, 1, 1)
        
        self.plot_losses_for_cameras(ax1, rest_of_cameras, "gray")
        self.plot_losses_for_cameras(ax1, best_cameras, "green", "Best")
        self.plot_losses_for_cameras(ax1, worst_cameras, "red", "Worst")
        ax1.legend()
        
        self.plot_images_for_cameras(fig, 4, best_cameras, "Best")
        self.plot_images_for_cameras(fig, 7, middle_cameras, "Mid")
        self.plot_images_for_cameras(fig, 10, worst_cameras, "Worst")
          
        plt.tight_layout()
        plt.savefig(f"{self.loss_dir}/camera_losses.png")
        plt.close()
        
    def create_visualizations(self):

        self.time_it("Loss vis", self.create_loss_visualization)
        self.time_it("Train vis", self.create_training_visualization)
        self.time_it("Xyz vis", self.fast_create_points_visualization, self.xyz_interval)
        self.time_it("3D vis", self.create_3D_visualization)
        self.time_it("Final Image", self.make_final_image)
        self.time_it("Gradients", self.visualize_grads)
        self.time_it("Camera_losses", self.save_loss_for_cameras)
        self.time_it("Results", self.save_results)
        self.time_it("History", self.save_history)
        print("Visualizations created")
