import viser
import viser.transforms as vtf
import numpy as onp
import time
import imageio.v3 as iio
import torch 
import torch.nn.functional as F
import torchvision
from typing import Dict
from src.occuthree import OccuGrid
from src.dataparser_outputs import Cameras
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians

class ViserViewer:
    def __init__(self, port, host = None, occugrid = None, 
                 cameras = None, images = None, bg_color = None,
                 grid = None):
        
        if host is not None:
            self.server = viser.ViserServer(port = port, host = host)
        else:
            self.server = viser.ViserServer(port = port)
        self.device = "cuda:0"
        #self.grid = OccuGrid(grid["resolution"],
        #                     grid["num_samples"],
        #                     device = self.device)
        
        self.grid = occugrid
        
        
        self.is_start = True
        self.is_paused = True
        self.is_finished = False
        self.is_recording = False
        
        self.grids = []
        self.images = []
        
        self.low_color = torch.Tensor([0, 0, 255]).to(self.device)
        self.high_color = torch.Tensor([255, 0, 0]).to(self.device)
        
        self.init_message = self.server.add_gui_text("Weight For Initialization",
                                                     initial_value="Weight For Initialization")
        self.start_button = self.server.add_gui_button("Start")
        self.pause_button = self.server.add_gui_button("Pause")
        self.resume_button = self.server.add_gui_button("Resume")
        self.start_record_button = self.server.add_gui_button("Start Recording")
        self.stop_record_button = self.server.add_gui_button("Stop Recording")
        self.download_file = self.server.add_gui_button("Download GIF")
        self.finished_button = self.server.add_gui_button("Finished")
        self.rgb_button = self.server.add_gui_button("RGB")
        self.opacity_button = self.server.add_gui_button("Opacity")
        self.combined_button = self.server.add_gui_button("Combined")
        self.scales_button = self.server.add_gui_button("Scales")
        self.snapshot_button = self.server.add_gui_button("Snap")
        self.stop_update_button = self.server.add_gui_button("Stop Update") 
        self.resume_update_button = self.server.add_gui_button("Resume Update")
        
        self.randomize_checkbox = self.server.add_gui_checkbox(
            "Randomize",
            initial_value = True
        )
        
        self.snapshot_button.visible = True       
        self.combined_button.visible = True
        self.scales_button.visible = True
        self.opacity_button.visible = True
        self.start_button.visible = False
        self.resume_button.visible = False
        self.pause_button.visible = False
        self.finished_button.visible = False
        self.start_record_button.visible = False
        self.stop_record_button.visible = False
        self.rgb_button.visible = False
        self.stop_update_button.visible = True
        self.resume_update_button.visible = False
        
        self.grid_visibility = self.server.add_gui_checkbox(
            "Grid",
            initial_value = False
        )
        
        self.gui_plane = self.server.add_gui_dropdown(
            "Plane",
            ("xy", "xz", "yz")
        ) 
        self.plane_dict = {
            "xz": lambda x: (0.5, x, 0.5),
            "yz": lambda x: (x, 0.5, 0.5),
            "xy": lambda x: (0.5, 0.5, x)
        }
        
        self.opacity_slider = self.server.add_gui_slider(
            "opacity_threshold",
            min = 0.0,
            max = 1.0,
            step = 0.01,
            initial_value = 0.5
        )
        
        self.scales_slider = self.server.add_gui_slider(
            "scale_threshold",
            min = 0.0,
            max = 1.0,
            step = 0.01,
            initial_value = 0.5
        )
        
        self.n_samples_slider = self.server.add_gui_slider(
            "n_samples",
            min = 1,
            max = 100000,
            step = 1,
            initial_value = self.grid.num_samples
        )

        self.grid_sliders = {
            "cell_thickness": self.add_slider_grid("cell_thickness", 0.1, 2.0, 0.1, 0.5),
            "section_thickness": self.add_slider_grid("section_thickness", 0.0, 1.0, 0.1, 0.0),
            "cell_size": self.add_slider_grid("cell_size", 0.1, 1.0, 0.1, 0.5),
        }
        self.point_size = self.server.add_gui_slider(
            "point_size",
            min = 0.001,
            max = 0.05,
            step = 0.001,
            initial_value = 0.02
        )
        self.update_interval = self.server.add_gui_slider(
            "update_interval",
            min = 0.01,
            max = 10,
            step = 0.01,
            initial_value = 3
        )

        self.take_snapshot = False
        @self.snapshot_button.on_click
        def _(_):
            self.take_snapshot = not self.take_snapshot        
        
        @self.start_button.on_click
        def _(_):
            if self.is_start:
                self.is_paused = False
                self.is_start = False
                self.start_button.visible = False
                self.pause_button.visible = True
            else:
                self.is_paused = False
                self.pause_button.visible = True
                self.resume_button.visible = False

        @self.pause_button.on_click
        def _(_):
            self.is_paused = True
            self.pause_button.visible = False
            self.resume_button.visible = True
            print("Training Paused")
            
        @self.resume_button.on_click
        def _(_):
            self.is_paused = False
            self.pause_button.visible = True
            self.resume_button.visible = False
        
        @self.grid_visibility.on_update
        def _(_):
            if self.grid_visibility.value == False:
                for grid in self.grids:
                    grid.remove()
            elif self.grid_visibility.value == True:
                self.create_3d_grid()
        
        @self.finished_button.on_click
        def _(_):
            self.is_finished = True
        
        @self.download_file.on_click
        def _(event) -> None:
            self.send_gif(event)
         
        @self.start_record_button.on_click
        def _(_):
            self.is_recording = not self.is_recording
            self.start_record_button.visible = False
            self.stop_record_button.visible = True
        @self.stop_record_button.on_click
        def _(_):
            self.is_recording = not self.is_recording
            self.stop_record_button.visible = False
            self.start_record_button.visible = True
        
        self.let_update = True
        @self.stop_update_button.on_click
        def _(_):
            self.let_update = False
            self.resume_update_button.visible = True
            self.stop_update_button.visible = False
            self.best_cameras = [[i, float("inf")] for i in range(3)]
            self.worst_cameras = [[i, float("-inf")] for i in range(3)]
            self.updated_best = [False for _ in range(3)]
            self.updated_worst = [False for _ in range(3)]
        @self.resume_update_button.on_click
        def _(_):
            self.let_update = not self.let_update    
            self.resume_update_button.visible = False
            self.stop_update_button.visible = True
        
        self.rgb_chosen = True
        self.opacities_chosen = False
        self.combined_chosen = False
        self.scales_chosen = False
        
        self.point_size.on_update(lambda _: self.create_3d_grid())
        self.gui_plane.on_update(lambda _: self.create_3d_grid())
        
        #if occugrid is not None:
        #    self.initialize_with_grid(occugrid)
        #else:
        self.initialize_temp()
        
        self.background_color = bg_color.to(self.device)
        self.cameras = cameras
        self.camera_handles: Dict[int,
                                  viser.CameraFrustumHandle,
                                  float,
                                  onp.array] = {}
        self.gt_images = images
        self.initialize_cameras()
        self.initialize_cameras_on_scene()
        self.init_message.visible = False
        self.start_button.visible = True
        self.start_record_button.visible = True
        self.best_cameras = [[i, float("inf")] for i in range(3)]   
        self.worst_cameras = [[i, float("-inf")] for i in range(3)]
        self.updated_best = [False for _ in range(3)]
        self.updated_worst = [False for _ in range(3)]
        
        @self.rgb_button.on_click
        def _(_):
            self.set_visibility('rgb', ['combined', 'opacity', 'scales'], ['rgb'])
        @self.opacity_button.on_click
        def _(_):
            self.set_visibility('opacities', ['rgb', 'scales', 'combined'], ['opacities'])
        @self.scales_button.on_click
        def _(_):
            self.set_visibility('scales', ['rgb', 'opacity', 'combined'], ['scales'])
        @self.combined_button.on_click
        def _(_):
            self.set_visibility('combined', ['rgb', 'opacity', 'scales'], ['combined'])    

        self.client = None 
        @self.server.on_client_connect
        def _(client):
            self.client = client
            @self.client.camera.on_update
            def _(_):
                if self.take_snapshot:
                    with self.client.atomic():
                        self.display_image(self.client)
                
        
        
        
    def set_visibility(self, chosen_button, visible_buttons, invisible_buttons):
        self.opacities_chosen = chosen_button == 'opacities'
        self.rgb_chosen = chosen_button == 'rgb'
        self.combined_chosen = chosen_button == 'combined'
        self.scales_chosen = chosen_button == 'scales'

        self.opacity_button.visible = 'opacity' in visible_buttons
        self.rgb_button.visible = 'rgb' in visible_buttons
        self.combined_button.visible = 'combined' in visible_buttons
        self.scales_button.visible = 'scales' in visible_buttons
        
        self.rgbs.visible = self.rgb_chosen
        self.opacities.visible = self.opacities_chosen
        self.combined.visible = self.combined_chosen
        self.scales.visible = self.scales_chosen
        
        for point_cloud in [self.rgbs, self.opacities, self.scales, self.combined]:
            if not point_cloud.visible:
                point_cloud.remove()
        
    def initialize_with_grid(self, grid):
        self.resolution = grid.resolution
        self.grid = grid
        self.initial_points = onp.array(self.grid.data)
        self.points = onp.array([[0,0,0]])
        self.add_points("/initial",self.initial_points, visibility = True, color = (0,255,0))
        
    def initialize_temp(self):
        self.resolution = 4
        self.points = onp.array([[0,0,0]])
        self.add_points("/initial",self.points,  visibility = True, color = (0,255,0))
        self.rgbs = self.add_points("/colors", self.points, visibility = False, color = (0,0,0))
        self.opacities = self.add_points("/opacities", self.points, visibility = False, color = (0,0,0))
        self.combined = self.add_points("/combined", self.points, visibility = False, color = (0,0,0))
        self.scales = self.add_points("/scales", self.points, visibility = False, color = (0,0,0))
        
    def add_slider_grid(self, name, min, max, step, initial_value):
        slider = self.server.add_gui_slider(
            name,
            min = min,
            max = max,
            step = step,
            initial_value = initial_value)
        slider.on_update(lambda _: self.create_3d_grid())
        return slider
    
    def add_points(self,node, points ,visibility, color = (0,0,0)):
        return self.server.add_point_cloud(
            f"/{node}",
            points = points,
            point_size = self.point_size.value,
            colors = color,
            point_shape = "circle",
            visible = visibility)
    
    def create_3d_grid(self):
        
        params = {name: slider.value for name, slider in self.grid_sliders.items()}

        self.server.add_point_cloud(
            "/center",
            points=onp.array([[0, 0, 0]]),
            point_size = self.point_size.value,
            colors = onp.array([[255, 0, 0]]),
            point_shape = "circle"
        )
        
        for plane, position in self.plane_dict.items():
            for i in range(self.resolution+1):
                self.grids.append(self.server.add_grid(f"/grid_{plane}_{i}", 
                                    width = 1,
                                    height = 1,
                                    position = position(i/self.resolution),
                                    plane = plane,
                                    **params))
    def send_gif(self, event):

        client = event.client
        print("Sending gif...")
        client.send_file_download(
            "created_gif.gif", iio.imwrite("<bytes>", self.images, extension=".gif")
        )
        print("Gif sent")

    def initialize_cameras(self):
        camera_idx = self.cameras.camera_idxs.detach().cpu().numpy().astype(int)
        camera_to_worlds = self.cameras.camera_to_worlds.detach().cpu().numpy()
        fx, fy = self.cameras.fx.detach().cpu().numpy(), self.cameras.fy.detach().cpu().numpy()
        cx, cy = self.cameras.cx.detach().cpu().numpy(), self.cameras.cy.detach().cpu().numpy()
        self.cameras = Cameras(
            camera_idxs = camera_idx,
            camera_to_worlds = camera_to_worlds,
            fx = fx,
            fy = fy,
            cx = cx,
            cy = cy
        )
        for idx, image in enumerate(self.gt_images):
            image = onp.array(image.cpu(), dtype = "uint8").astype("float32") / 255.0
            image = torch.from_numpy(image)[:,:,:3]
            #image = image[:, :, :3] * image[:, :, -1:] + self.background_color.cpu() * (
            #    1.0 - image[:, :, -1:]) 
            image_uint8 = (image * 255).detach().type(torch.uint8).to("cuda:0")
            image_uint8 = image_uint8.permute(2, 0 , 1)
            image_uint8 = torchvision.transforms.functional.resize(image_uint8, 100, antialias = None)
            image_uint8 = image_uint8.permute(1, 2, 0)
            self.gt_images[idx] = image.cpu().numpy()
        
        
    def initialize_cameras_on_scene(self):
        H, W = self.cameras.cx * 2, self.cameras.cy * 2
        fy = self.cameras.fy
        for camera_idx in self.cameras.camera_idxs:
            c2w  = self.cameras.camera_to_worlds[camera_idx]
            R = vtf.SO3.from_matrix(c2w[:3, :3])
            R = R @ vtf.SO3.from_x_radians(onp.pi)
            
            camera_handle = self.server.add_camera_frustum(
              f"/cameras/camera_{camera_idx}",
              fov = float(2 * onp.arctan(self.cameras.cx / self.cameras.fx)),
              aspect = float(W / H),
              scale = 0.15,
              wxyz = R.wxyz,
              position = c2w[:3, 3],
              image = self.gt_images[camera_idx]
            )
            @camera_handle.on_click
            def _(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]) -> None:
                with event.client.atomic():
                    event.client.camera.position = event.target.position
                    event.client.camera.wxyz = event.target.wxyz
            self.camera_handles[camera_idx] = {"cam_handle": camera_handle,
                                               "loss": 1000,
                                               "image": None}  
    
    def subsample_points(self, num_points, *point_tensors):
        indices = torch.randperm(point_tensors[0].shape[0])[:num_points]
        return [point_tensor[indices] for point_tensor in point_tensors]
            
    def update(self, loss, pred_img,
                camera_idx,
                means, rgb, opacities, scales,
                model):
        while self.is_paused:
            time.sleep(0.1)
        if self.let_update:
            
            start = time.time()
            self.clients = self.server.get_clients()
            #means = onp.array(means.detach().cpu())
            self.model = model
            self.update_losses(camera_idx, loss, pred_img)
            self.update_camera_images(camera_idx, pred_img)
            self.update_camera_colors()
            
            if self.opacities_chosen:
                self.display_opacities(means, opacities)
                
            elif self.rgb_chosen:
                self.display_rgb(means, rgb)
                
            elif self.combined_chosen:
                self.display_combined(means, rgb, opacities)
                
            elif self.scales_chosen:
                self.display_scales(means, scales)
            
            if self.is_recording:
                self.record_screen(self.clients)
            
            for i, client in self.clients.items():
                    self.client = client 
            
            #end = time.time() - start
            #print(f"Time elapsed: {end} s, {end/60} min, fps: {1/end}") 
            time.sleep(self.update_interval.value)
        
    def quaternion_to_forward_vector(self, q):
        # Quaternion multiplication with the vector (0, 0, -1)
        t1 = 2.0 * (q[1] * q[3] - q[0] * q[2])
        t2 = 2.0 * (q[2] * q[3] + q[0] * q[1])
        t3 = -1.0 + 2.0 * (q[0] * q[0] + q[1] * q[1])

        return onp.array([t1, t2, t3])

    
    def update_client_image(self, client):
        self.image_cam_handle = client.camera
        
        @self.image_cam_handle.on_update
        def _(_):
            with client.atomic():
                if self.take_snapshot:
                    self.display_image(client)
        
    
    def display_image(self, client):
        model = self.model
        #for i, client in client.items():
        model.eval()
        
        with torch.no_grad():
            self.grid.num_samples = self.n_samples_slider.value
            #xyz = self.grid.sample(randomize = self.randomize_checkbox.value)
            xyz = self.grid.importance_sample(randomize = self.randomize_checkbox.value)
            xyz = self.grid.reverse_normalize(xyz)
            
            quats, rgbs, opacities , scales = model(xyz)
        image = self.create_image(client, xyz, quats, rgbs, opacities, scales)
        
        #normalize forward vector with numpy
        
        f_v = client.camera.position - client.camera.look_at
        norm = onp.linalg.norm(f_v) 
        f_v =  1.2 * (f_v / norm)
        #offset_to_front = 0.9 * f_v
        
        #print(offset_to_front)
        self.client.add_image("/image",
                            image.detach().cpu().numpy(),
                            position = client.camera.position - f_v,
                            wxyz = client.camera.wxyz,
                            render_width=0.6,
                            render_height=0.6)
     
    def create_image(self, client, means, quats, rgbs, opacities, scales):
        with client.atomic():
            background = self.background_color
            R = vtf.SO3(wxyz = client.camera.wxyz)
            R = R @ vtf.SO3.from_x_radians(onp.pi)
            R = torch.tensor(R.as_matrix()).to(self.device)
            R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
            pos = torch.tensor(client.camera.position,
                            dtype = torch.float32, device = self.device)
            c2w = torch.concatenate([R, pos[:, None]], dim = 1)
            viewmat = torch.eye(4, dtype = torch.float32, device = self.device)
            R = c2w[:3, :3]
            T = c2w[:3, 3:4]
            R = R @ R_edit
            R_inv = R.T
            T_inv = -R_inv @ T
            viewmat[:3, :3] = R_inv
            viewmat[:3, 3:4] = T_inv
            viewmat = viewmat
            H, W = int(self.cameras.cx * 2), torch.Tensor(self.cameras.cx * 2)
            focal_length = torch.Tensor(self.cameras.fx).to(self.device)

            xys, depths, radii, conics, comp, n_tiles_hit, cov3d = project_gaussians(
                means,
                scales,
                1, 
                quats,
                viewmat,
                focal_length,
                focal_length,
                H / 2 ,
                W / 2,
                H, 
                W,
                16
            )

            out_img = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                n_tiles_hit,
                rgbs,
                opacities,
                H,
                W,
                16,
                background
            )
            return out_img
        
        
    def display_rgb(self, means, rgb):

        means = means.detach().cpu().numpy()
        rgb = rgb.detach().cpu().numpy()
        self.rgbs = self.add_points("/colors",
                        points = means,
                        visibility=self.rgb_chosen,
                        color = rgb)
    
    def display_opacities(self, means, opacities):
        means, opacities = self.subsample_points(10000, means, opacities)
        mask = opacities > self.opacity_slider.value
        
        opacities = opacities[mask[:, 0]]
        means = means[mask[:, 0]]
        colors = (1 - opacities) * self.low_color + opacities * self.high_color
        
        means = means.detach().cpu().numpy()
        colors = colors.detach().cpu().numpy()
        
        self.opacities = self.add_points("/opacities",
                        points = means,
                        visibility = self.opacities_chosen,
                        color = colors
                        )
        
    def display_combined(self, means, rgb, opacities):
        colors = (1 - opacities) * self.background_color + opacities * rgb

        mask = opacities > self.opacity_slider.value
        colors = colors[mask[:, 0]]
        means = means[mask[:, 0]]
        
        means = means.detach().cpu().numpy()
        colors = colors.detach().cpu().numpy()
        
        self.combined = self.add_points("/combined",
                                    points = means,
                                    visibility = self.combined_chosen,
                                    color = colors)
    
    def display_scales(self, means, scales):

        scales = torch.max(scales, dim=-1).values
        scales = (scales - scales.min()) / (scales.max() - scales.min())
        mask = scales > self.scales_slider.value
        means = means[mask]
        scales = scales[mask].view(-1, 1)
        colors = (1 - scales) * self.low_color + scales * self.high_color
        
        means = means.detach().cpu().numpy()
        colors = colors.detach().cpu().numpy()
        self.scales = self.add_points("/scales",
                                points = means,
                                visibility = self.scales_chosen,
                                color = colors)
    
    def update_camera_colors(self):
        for cam_idx, _ in self.best_cameras:
            self.camera_handles[cam_idx]["cam_handle"] = self.add_camera(cam_idx, 
                                                                         color = (0, 255, 0), 
                                                                         image = self.camera_handles[cam_idx]["image"])
        for cam_idx, _ in self.worst_cameras:
            self.camera_handles[cam_idx]["cam_handle"] = self.add_camera(cam_idx, 
                                                                         color = (255, 0, 0), 
                                                                         image = self.camera_handles[cam_idx]["image"])
        
    def update_losses(self, camera_idx, loss, pred_img):
        with torch.no_grad():

            curr_camera_idx = int(camera_idx.item())
            self.camera_handles[curr_camera_idx]["loss"] = loss
            
            for i, (old_cam_idx, _) in enumerate(self.best_cameras):
                if loss < self.best_cameras[i][1]:
                    #replace the camera with a better one
                    self.best_cameras[i] = [curr_camera_idx, loss]
                    self.camera_handles[curr_camera_idx]["image"] = pred_img.detach().cpu().numpy()
                    #change the old camera color to black
                    self.camera_handles[old_cam_idx]["cam_handle"] = self.add_camera(old_cam_idx, color = (20, 20, 20))
                    self.updated_best[i] = True
                    break
            for i, (old_cam_idx, _) in enumerate(self.worst_cameras):
                if loss > self.worst_cameras[i][1]:
                    #replace the camera with a better one
                    self.worst_cameras[i] = [curr_camera_idx, loss]
                    self.camera_handles[curr_camera_idx]["image"] = pred_img.detach().cpu().numpy()
                    #change the old camera color to black
                    self.camera_handles[old_cam_idx]["cam_handle"] = self.add_camera(old_cam_idx, color = (20, 20, 20))
                    self.updated_worst[i] = True
                    break
    
    def update_camera_images(self, camera_idx, pred_img):
        current_camera_idx = int(camera_idx.item())
        for i, updated in enumerate(self.updated_best):
            camera_idx_to_be_updated = self.best_cameras[i][0]
            if not updated and current_camera_idx == camera_idx_to_be_updated:
                self.camera_handles[current_camera_idx]["image"] = pred_img.detach().cpu().numpy()
            self.updated_best[i] = False
            
        for i, updated in enumerate(self.updated_worst):
            camera_idx_to_be_updated = self.worst_cameras[i][0]
            if not updated and current_camera_idx == camera_idx_to_be_updated:
                self.camera_handles[current_camera_idx]["image"] = pred_img.detach().cpu().numpy()
            self.updated_worst[i] = False
            
    def add_camera(self, camera_idx, color = (20, 20, 20), image = None):
        if image is None:
            image = self.gt_images[camera_idx]
        else:
            image = image
        H, W = self.cameras.cx * 2, self.cameras.cy * 2
        c2w  = self.cameras.camera_to_worlds[camera_idx]
        R = vtf.SO3.from_matrix(c2w[:3, :3])
        R = R @ vtf.SO3.from_x_radians(onp.pi)
        
        camera_handle = self.server.add_camera_frustum(
            f"/cameras/camera_{camera_idx}",
            fov = float(2 * onp.arctan(self.cameras.cx / self.cameras.fx)),
            aspect = float(W / H),
            scale = 0.15,
            wxyz = R.wxyz,
            position = c2w[:3, 3],
            image = image,
            color = color
        )
        @camera_handle.on_click
        def _(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]) -> None:
            with event.client.atomic():
                event.client.camera.position = event.target.position
                event.client.camera.wxyz = event.target.wxyz
        return camera_handle

    def record_screen(self, clients):
        for i, client in clients.items():
            print(f"Sending to client {i}")
            self.images.append(client.camera.get_render(height=720, width=1280))
            
    def idle(self):
        self.resume_button.visible = False
        self.pause_button.visible = False
        self.finished_button.visible = True
        while not self.is_finished:
            time.sleep(0.1)
            
            
            
            
            
            
                
                