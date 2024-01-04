import torch
from torch import nn
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from urllib.request import urlretrieve
import pickle
import os
import math

apartments = ['apartment_1', 'apartment_2', 'frl_apartment_2', 'frl_apartment_4', 'office_4', 'room_2']
apt_idx = 2
apt = apartments[apt_idx]
print("Selected room name: {}".format(apt))

metadata_path = "https://www.andrew.cmu.edu/user/afluo/required_metadata/"

meanstd_path = metadata_path + r"mean_std/"+apt+".pkl"
weight_path = metadata_path + r"weights/"+apt+r"/00200.chkpt"
minmax_path = metadata_path + r"minmax/"+apt+"_minmax.pkl"
grid_path = metadata_path + r"room_grid_coors/"+apt+".pkl"

def download(url_path, out_path):
    return urlretrieve(url_path, out_path)

def load_pkl(path):
    with open(path, "rb") as loaded_pkl_obj:
        loaded_pkl = pickle.load(loaded_pkl_obj)
    return loaded_pkl

meanstd_path_out = "meanstd_"+apt
weight_path_out = "weight_"+apt
minmax_path_out = "minmax_"+apt
grid_path_out = "grid_"+apt

print("Downloading normalization metadata")
print(download(meanstd_path, meanstd_path_out))

print("Downloading network weights")
print(download(weight_path, weight_path_out))

print("Downloading room bbox")
print(download(minmax_path, minmax_path_out))

print("Downloading room locations")
print(download(grid_path, grid_path_out))

grid_coors = np.array(list(load_pkl(grid_path_out).values()))
plt.scatter(grid_coors[:,0], grid_coors[:,1])
plt.title("Top down view of {}".format(apt))
plt.axis('equal')
plt.show()

np.random.seed(1)
rand_pt_idx = np.random.choice(len(grid_coors))
rand_pt = grid_coors[rand_pt_idx]


np.random.seed(2)
rand_pt_idx_2 = np.random.choice(len(grid_coors))
rand_pt_2 = grid_coors[rand_pt_idx_2]
print("Two random point are: ", rand_pt, rand_pt_2)
plt.scatter(grid_coors[:,0], grid_coors[:,1])
plt.scatter(rand_pt[0], rand_pt[1], c="red")
plt.scatter(rand_pt_2[0], rand_pt_2[1], c="yellow")
plt.title("Selected point shown in red and yellow")
plt.axis('equal')
plt.show()

mean_std = load_pkl(meanstd_path_out)
print("Loaded mean std")
mean = torch.from_numpy(mean_std[0]).float()[None]
std = 3.0 * torch.from_numpy(mean_std[1]).float()[None]

class embedding_module_log(nn.Module):
    def __init__(self, funcs=[torch.sin, torch.cos], num_freqs=20, max_freq=10, ch_dim=1, include_in=True):
        super().__init__()
        self.functions = funcs
        self.num_functions = list(range(len(funcs)))
        self.freqs = torch.nn.Parameter(2.0**torch.from_numpy(np.linspace(start=0.0,stop=max_freq, num=num_freqs).astype(np.single)), requires_grad=False)
        self.ch_dim = ch_dim
        self.funcs = funcs
        self.include_in = include_in

    def forward(self, x_input):
        if self.include_in:
            out_list = [x_input]
        else:
            out_list = []
        for func in self.funcs:
            for freq in self.freqs:
                out_list.append(func(x_input*freq))
        return torch.cat(out_list, dim=self.ch_dim)

def distance(x1, x2):
    # by jacobrgardner
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    return res

def fit_predict_torch(input_pos:torch.Tensor, input_target:torch.Tensor, predict_pos:torch.Tensor, bandwidth:torch.Tensor) -> torch.Tensor:
    dist_vector = -distance(predict_pos, input_pos)
    gauss_dist = torch.exp(dist_vector/(2.0 * torch.square(bandwidth.unsqueeze(0))))
    magnitude = torch.sum(gauss_dist, dim=1, keepdim=True)
    out = torch.mm(gauss_dist, input_target)/magnitude
    return out

class basic_project2(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(basic_project2, self).__init__()
        self.proj = nn.Linear(input_ch, output_ch, bias=True)
    def forward(self, x):
        return self.proj(x)

class kernel_linear_act(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(kernel_linear_act, self).__init__()
        self.block = nn.Sequential(nn.LeakyReLU(negative_slope=0.1), basic_project2(input_ch, output_ch))
    def forward(self, input_x):
        return self.block(input_x)

class kernel_residual_fc_embeds(nn.Module):
    def __init__(self, input_ch, intermediate_ch=512, grid_ch = 64, num_block=8, output_ch=1, grid_gap=0.25, grid_bandwidth=0.25, bandwidth_min=0.1, bandwidth_max=0.5, float_amt=0.1, min_xy=None, max_xy=None, probe=False):
        super(kernel_residual_fc_embeds, self).__init__()
        # input_ch (int): number of ch going into the network
        # intermediate_ch (int): number of intermediate neurons
        # min_xy, max_xy are the bounding box of the room in real (not normalized) coordinates
        # probe = True returns the features of the last layer

        for k in range(num_block - 1):
            self.register_parameter("left_right_{}".format(k),nn.Parameter(torch.randn(1, 1, 2, intermediate_ch)/math.sqrt(intermediate_ch),requires_grad=True))

        for k in range(4):
            self.register_parameter("rot_{}".format(k), nn.Parameter(torch.randn(num_block - 1, 1, 1, intermediate_ch)/math.sqrt(intermediate_ch), requires_grad=True))

        self.proj = basic_project2(input_ch + int(2*grid_ch), intermediate_ch)
        self.residual_1 = nn.Sequential(basic_project2(input_ch + 128, intermediate_ch), nn.LeakyReLU(negative_slope=0.1), basic_project2(intermediate_ch, intermediate_ch))
        self.layers = torch.nn.ModuleList()
        for k in range(num_block - 2):
            self.layers.append(kernel_linear_act(intermediate_ch, intermediate_ch))

        self.out_layer = nn.Linear(intermediate_ch, output_ch)
        self.blocks = len(self.layers)
        self.probe = probe

        ### Make the grid

        grid_coors_x = np.arange(min_xy[0], max_xy[0], grid_gap)
        grid_coors_y = np.arange(min_xy[1], max_xy[1], grid_gap)
        grid_coors_x, grid_coors_y = np.meshgrid(grid_coors_x, grid_coors_y)
        grid_coors_x = grid_coors_x.flatten()
        grid_coors_y = grid_coors_y.flatten()
        xy_train = np.array([grid_coors_x, grid_coors_y]).T
        self.bandwidth_min = bandwidth_min
        self.bandwidth_max = bandwidth_max
        self.float_amt = float_amt
        self.bandwidths = nn.Parameter(torch.zeros(len(grid_coors_x))+grid_bandwidth, requires_grad=True)
        self.register_buffer("grid_coors_xy",torch.from_numpy(xy_train).float(), persistent=True)
        self.xy_offset = nn.Parameter(torch.zeros_like(self.grid_coors_xy), requires_grad=True)
        self.grid_0 = nn.Parameter(torch.randn(len(grid_coors_x),grid_ch, device="cpu").float() / np.sqrt(float(grid_ch)), requires_grad=True)

    def forward(self, input_stuff, rot_idx, sound_loc=None):
        SAMPLES = input_stuff.shape[1]
        sound_loc_v0 = sound_loc[..., :2]
        sound_loc_v1 = sound_loc[..., 2:]

        # Prevent numerical issues
        self.bandwidths.data = torch.clamp(self.bandwidths.data, self.bandwidth_min, self.bandwidth_max)

        grid_coors_baseline = self.grid_coors_xy + torch.tanh(self.xy_offset) * self.float_amt
        grid_feat_v0 = fit_predict_torch(grid_coors_baseline, self.grid_0, sound_loc_v0, self.bandwidths)
        grid_feat_v1 = fit_predict_torch(grid_coors_baseline, self.grid_0, sound_loc_v1, self.bandwidths)
        total_grid = torch.cat((grid_feat_v0, grid_feat_v1), dim=-1).unsqueeze(1).expand(-1, SAMPLES, -1)

        my_input = torch.cat((total_grid, input_stuff), dim=-1)
        rot_latent = torch.stack([getattr(self, "rot_{}".format(rot_idx_single)) for rot_idx_single in rot_idx], dim=0)
        out = self.proj(my_input).unsqueeze(2).repeat(1, 1, 2, 1) + getattr(self, "left_right_0") + rot_latent[:, 0]
        for k in range(len(self.layers)):
            out = self.layers[k](out) + getattr(self, "left_right_{}".format(k + 1)) + rot_latent[:, k + 1]
            if k == (self.blocks // 2 - 1):
                out = out + self.residual_1(my_input).unsqueeze(2).repeat(1, 1, 2, 1)
        if self.probe:
            return out
        return self.out_layer(out)


weights = torch.load(weight_path_out, map_location='cuda:0')
min_maxes = load_pkl(minmax_path_out)
print("Bounding box of the room: ", min_maxes)
min_pos = min_maxes[0][[0, 2]]
max_pos = min_maxes[1][[0, 2]]
max_lengths = {"apartment_1": 101, "apartment_2": 86, "frl_apartment_2": 107, "frl_apartment_4": 103,
                   "office_4": 78, "room_2": 84}
# The maximum length of an impulse response, derived from data
output_device = 0
num_freqs = 10

xyz_embedder = embedding_module_log(num_freqs=num_freqs, ch_dim=2, max_freq=7).to(output_device)
time_embedder = embedding_module_log(num_freqs=num_freqs, ch_dim=2).to(output_device)
freq_embedder = embedding_module_log(num_freqs=num_freqs, ch_dim=2).to(output_device)
auditory_net = kernel_residual_fc_embeds(input_ch=126, intermediate_ch=512, grid_ch=64, num_block=8, grid_gap=0.25,
                                             grid_bandwidth=0.25, bandwidth_min=0.1, bandwidth_max=0.5, float_amt=0.1,
                                             min_xy=min_pos, max_xy=max_pos)
loaded = auditory_net.load_state_dict(weights["network"])
loaded = auditory_net.to("cuda:0")


def prepare_input(orientation_idx, reciever_pos, source_pos, max_len, min_bbox_pos, max_bbox_pos):
    selected_time = np.arange(0, max_len)
    selected_freq = np.arange(0, 256)
    selected_time, selected_freq = np.meshgrid(selected_time, selected_freq)
    selected_time = selected_time.reshape(-1)
    selected_freq = selected_freq.reshape(-1)

    degree = orientation_idx

    non_norm_start = np.array(reciever_pos)
    non_norm_end = np.array(source_pos)
    total_non_norm_position = torch.cat((torch.from_numpy(non_norm_start)[None], torch.from_numpy(non_norm_end)[None]), dim=1).float()

    start_position = (torch.from_numpy((non_norm_start - min_bbox_pos) / (max_bbox_pos - min_bbox_pos))[None] - 0.5) * 2.0
    start_position = torch.clamp(start_position, min=-1.0, max=1.0)
    end_position = (torch.from_numpy((non_norm_end - min_bbox_pos) / (max_bbox_pos - min_bbox_pos))[None] - 0.5) * 2.0
    end_position = torch.clamp(end_position, min=-1.0, max=1.0)
    total_position = torch.cat((start_position, end_position), dim=1).float()

    return degree, total_position, total_non_norm_position, 2.0*torch.from_numpy(selected_freq).float()/255.0 - 1.0, 2.0*torch.from_numpy(selected_time).float()/float(max_len-1)-1.0

emitter_position = rand_pt_2
listener_position = rand_pt
orientation = 0 # can be of 0, 1, 2, 3. Corresponds to 0, 90, 180, 270 degrees rotation for the listener

transformed_input = prepare_input(0, listener_position, emitter_position, max_lengths[apt], min_pos, max_pos)

# Now we apply sinusoidal embeddings:
degree = torch.Tensor([transformed_input[0]]).to(output_device, non_blocking=True).long()
position = transformed_input[1][None].to(output_device, non_blocking=True)
non_norm_position = transformed_input[2].to(output_device, non_blocking=True)
freqs = transformed_input[3][None].to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * math.pi
times = transformed_input[4][None].to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * math.pi
PIXEL_COUNT = max_lengths[apt] * 256
position_embed = xyz_embedder(position).expand(-1, PIXEL_COUNT, -1)
freq_embed = freq_embedder(freqs)
time_embed = time_embedder(times)
total_in = torch.cat((position_embed, freq_embed, time_embed), dim=2)

output_container = []
auditory_net.eval()
with torch.no_grad():
    output = auditory_net(total_in, degree, non_norm_position.squeeze(1)).squeeze(3).transpose(1, 2)
output = (output.reshape(1, 2, 256, max_lengths[apt]).cpu() * std[None] + mean[None]).numpy()
print("Completed inference")

fig, axarr = plt.subplots(1, 2)
fig.suptitle("Predicted log impulse response", fontsize=16)
axarr[0].imshow(output[0,0], cmap="inferno", vmin=np.min(output)*1.1, vmax= np.max(output)*0.9)
axarr[0].set_title('Channel 1')
axarr[0].axis("off")
# plt.subplot(1, 2, 2)
axarr[1].imshow(output[0,1], cmap="inferno", vmin=np.min(output)*1.1, vmax= np.max(output)*0.9)
axarr[1].set_title('Channel 2')
axarr[1].axis("off")
plt.show()

emitter_location = rand_pt_2
output_container = []
print("Total {} points to query".format(len(grid_coors)))

with torch.no_grad():
    for recv_loc_idx in range(len(grid_coors)):
        if recv_loc_idx%100 == 0:
            print("Currently on {}".format(recv_loc_idx))
        rec_loc = grid_coors[recv_loc_idx]
        transformed_input = prepare_input(0, rec_loc, emitter_location, max_lengths[apt], min_pos, max_pos)
        degree = torch.Tensor([transformed_input[0]]).to(output_device, non_blocking=True).long()
        position = transformed_input[1][None].to(output_device, non_blocking=True)
        non_norm_position = transformed_input[2].to(output_device, non_blocking=True)
        freqs = transformed_input[3][None].to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * math.pi
        times = transformed_input[4][None].to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * math.pi
        PIXEL_COUNT = max_lengths[apt] * 256
        position_embed = xyz_embedder(position).expand(-1, PIXEL_COUNT, -1)
        freq_embed = freq_embedder(freqs)
        time_embed = time_embedder(times)
        total_in = torch.cat((position_embed, freq_embed, time_embed), dim=2)
        output_container.append((auditory_net(total_in, degree, non_norm_position.squeeze(1)).squeeze(3).transpose(1, 2).reshape(1, 2, 256, max_lengths[apt]).cpu() * std[None] + mean[None]).numpy())

with open("loudness.pkl", "wb") as f:
    pickle.dump(output_container, f)

loudness_container = []
for i in range(len(output_container)):
    myout = np.clip(np.exp(output_container[i])-1e-3, 0.0, 10000.0)
    loudness_container.append(librosa.feature.rms(S=myout[0,0], frame_length=myout[0,0].shape[-2] * 2 - 2)+librosa.feature.rms(S=myout[0,1], frame_length=myout[0,1].shape[-2] * 2 - 2))
loudness = np.array(loudness_container)
plt.rcParams['figure.dpi'] = 90
plt.title("Loudness map for {} for emitter at red dot".format(apt))
plt.scatter(grid_coors[:,0], grid_coors[:,1], c=np.log(np.sum(loudness, axis=(1, 2))))
plt.scatter(emitter_location[0], emitter_location[1], c="red")
plt.axis('equal')
plt.axis('off')
plt.show()


def gen_trajectory(way_points, rotations, time_total=50, rotate_time=10, frame_rate=12):
    distances = []
    total_frames = float(time_total * frame_rate)
    orig_total = total_frames + 0.0
    cumulative_diff = np.abs(way_points[:-1] - way_points[1:])
    cumulative_diff = np.linalg.norm(cumulative_diff, axis=1) < 0.01

    # if the next waypoint is the same location as the current waypoint
    # We just rotate in place. For this tutorial we don't use this aspect
    way_points = np.array(way_points)
    for k in range(0, len(way_points) - 1):
        distances.append(np.linalg.norm(way_points[k + 1] - way_points[k]))
    total_distance = np.sum(distances)

    for k in range(0, len(way_points) - 1):
        if cumulative_diff[k]:
            total_frames = total_frames - int(
                np.rint((frame_rate * rotate_time) * (np.abs(rotations[k + 1] - rotations[k])) / 360.0))

    num_frames = []
    for k in range(0, len(way_points) - 1):
        if not cumulative_diff[k]:
            num_frames.append(int(np.rint(total_frames * distances[k] / total_distance)))
        else:
            print(rotations[k + 1] - rotations[k], "FRAMES")
            num_frames.append(
                int(np.rint(float(frame_rate * rotate_time) * (np.abs(rotations[k + 1] - rotations[k])) / 360.0)))
    num_frames[-1] = int(orig_total - np.sum(num_frames[:-1]))
    num_frames = [int(zzz) for zzz in num_frames]
    assert np.sum(num_frames) == orig_total

    frame_pieces_pos = []
    frame_pieces_rot = []
    for way_piece_idx in range(0, len(way_points) - 1):
        buffer_pos = []
        buffer_rot = []

        piece_frames = num_frames[way_piece_idx]
        if not piece_frames > 0:
            print("not enough frames!")
            continue
        for interp_idx in range(piece_frames):
            cur_position = way_points[way_piece_idx] * (1 - float(interp_idx) / float(piece_frames - 1)) + way_points[
                way_piece_idx + 1] * (float(interp_idx) / float(piece_frames - 1))
            buffer_pos.append(cur_position)

            cur_rot = rotations[way_piece_idx] * (1 - float(interp_idx) / float(piece_frames - 1)) + rotations[
                way_piece_idx + 1] * (float(interp_idx) / float(piece_frames - 1))

            buffer_rot.append(cur_rot)

        frame_pieces_pos.extend(buffer_pos)
        frame_pieces_rot.extend(buffer_rot)
    frame_pieces_pos = np.array(frame_pieces_pos)
    frame_pieces_rot = np.array(frame_pieces_rot)
    return frame_pieces_pos, frame_pieces_rot


way_pts = np.array([[-2.0, 0.0, -0.5], [2.0, 0.0, -0.5], [2.0, 0.0, -4.0], [8.0, 0.0, -4.0]])
way_rots = np.array([0.0]*5)
traj_points, traj_rots = gen_trajectory(way_pts, way_rots)
plt.scatter(grid_coors[:,0], grid_coors[:,1], c=np.log(np.sum(loudness, axis=(1, 2))))
plt.scatter(emitter_location[0], emitter_location[1], c="red")

trajectory_color = np.linspace(0,1,len(traj_points))
trajectory_color = np.stack([trajectory_color, trajectory_color, trajectory_color], axis=-1)
plt.scatter(traj_points[:,0],traj_points[:,-1],c=trajectory_color)
plt.scatter(traj_points[0,0], traj_points[0,-1], c="green")
plt.scatter(traj_points[-1,0], traj_points[-1,-1], c="blue")
plt.title("Trajectory start shown in green, end shown in blue")
plt.axis('equal')
plt.axis('off')
plt.show()

emitter_location = rand_pt_2
impulse_container = []
traj_points_query = traj_points[:,[0, 2]]
print("Total {} points to query".format(len(traj_points_query)))

with torch.no_grad():
    for recv_loc_idx in range(len(traj_points_query)):
        if recv_loc_idx%100 == 0:
            print("Currently on {}".format(recv_loc_idx))
        rec_loc = traj_points_query[recv_loc_idx]
        transformed_input = prepare_input(0, rec_loc, emitter_location, max_lengths[apt], min_pos, max_pos)
        degree = torch.Tensor([transformed_input[0]]).to(output_device, non_blocking=True).long()
        position = transformed_input[1][None].to(output_device, non_blocking=True)
        non_norm_position = transformed_input[2].to(output_device, non_blocking=True)
        freqs = transformed_input[3][None].to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * math.pi
        times = transformed_input[4][None].to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * math.pi
        PIXEL_COUNT = max_lengths[apt] * 256
        position_embed = xyz_embedder(position).expand(-1, PIXEL_COUNT, -1)
        freq_embed = freq_embedder(freqs)
        time_embed = time_embedder(times)
        total_in = torch.cat((position_embed, freq_embed, time_embed), dim=2)
        impulse_container.append((auditory_net(total_in, degree, non_norm_position.squeeze(1)).squeeze(3).transpose(1, 2).reshape(1, 2, 256, max_lengths[apt]).cpu() * std[None] + mean[None]).numpy())


def to_wave(input_spec, mean_val=None, std_val=None, gl=False, orig_phase=None):
    if not mean_val is None:
        renorm_input = input_spec * std_val
        renorm_input = renorm_input + mean_val
    else:
        renorm_input = input_spec + 0.0
    renorm_input = np.exp(renorm_input) - 1e-3
    renorm_input = np.clip(renorm_input, 0.0, 100000.0)
    if orig_phase is None:
        if gl == False:

            # Random phase reconstruction per image2reverb
            # do not use griffinlim
            np.random.seed(1234)
            rp = np.random.uniform(-np.pi, np.pi, renorm_input.shape)
            f = renorm_input * (np.cos(rp) + (1.j * np.sin(rp)))
            out_wave = librosa.istft(f, hop_length=128)
        else:
            out_wave = librosa.griffinlim(renorm_input, hop_length=128, n_iter=40, momentum=0.5, random_state=64)
    else:
        f = renorm_input * (np.cos(orig_phase) + (1.j * np.sin(orig_phase)))
        # out_wave = librosa.istft(f, win_length=400, hop_length=200)
    return out_wave


wave_container = []
for data in impulse_container:
    wave_container.append(to_wave(data))

tmp = wave_container[0].reshape(2, 13568)
plt.plot(tmp[0, :])
plt.show()
