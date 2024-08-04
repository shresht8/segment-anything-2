import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import matplotlib
matplotlib.use('TkAgg')

import io

# using bflot16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Change to any other checkpoint
sam2_checkpoint = "C:\\Users\\shres\\Projects\\segment-anything-2\\checkpoints\\sam2_hiera_large.pt"

# Change according to used checkpoint - in the sam2_configs dir
model_cfg = "sam2_hiera_l.yaml"

# create the predictor object
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)


# mask function
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


# function to show points
def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


# dir for video frames
video_dir = "C:\\Users\\shres\\Projects\\segment-anything-2\\video\\test2"

# scan all the jpeg frames in the dir
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg",".jpeg",".JPG",".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# take a look at the first frame (or any other) and find the object points
# use at least 2 points
frame_idx = 0 # use the frame number to be used
plt.figure(figsize=(12,8))
plt.title(f"frame {frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
plt.show()

print("Press enter to continue...")
input()
print("Continuing...")

inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)

ann_frame_idx = 0 # the frame index we are interacting with
ann_obj_id = 1 # giving a unique id to the object we want to interact with (any int can be given)

# Enter the point co-ords [x,y] as rows of a numpy array

points = np.array([[848,589],[955,609],[900,597]],dtype=np.float32) # replace with the co-ordinates of the chosen points
# for labels, 1 means pos click and 0 means neg click
labels = np.array([1, 1, 1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(12,8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
show_mask((out_mask_logits[0]>0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
plt.show()

print("Press enter to continue...")
input()
print("Continuing...")

# run propagation through the video and collect the results in a dict
video_segments = {} # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
vis_frame_stride = 1
plt.close("all")

# To add new points to refine the Object detection
ann_frame_idx = 8  # further refine some details on this frame
ann_obj_id = 1  # give a unique id to the object we interact with (it can be any integers)
plt.title(f"frame {ann_frame_idx} -- before refinement")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_mask(video_segments[ann_frame_idx][ann_obj_id], plt.gca(), obj_id=out_obj_ids[0])
plt.show()

print("Press enter to continue...")
input()
print("Continuing...")

points = np.array([[535,495],[613,530]],dtype=np.float32) # replace with the co-ordinates of the chosen points
# for labels, 1 means pos click and 0 means neg click
labels = np.array([0,0], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)


# show the segment after the further refinement
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx} -- after refinement")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
show_mask((out_mask_logits > 0.0).cpu().numpy(), plt.gca(), obj_id=ann_obj_id)
plt.show()

print("Press enter to continue...")
input()
print("Continuing...")

# run propagation throughout the video
video_segments = {} # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# segment multiple objects simultaeneously
print("Setting up for multi object detection")
predictor.reset_state(inference_state)
prompts = {}  # hold all the clicks we add for visualization

# add the first object
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)

# Let's add a 2nd negative click at (x, y) = (275, 175) to refine the first object
# sending all clicks (and their labels) to `add_new_points`
points = np.array([[957, 609], [555, 503]], dtype=np.float32) # change to points for first object
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 0], np.int32) # change to labels for first object
prompts[ann_obj_id] = points, labels
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.show()

print("Press enter to continue to chose the second object to detect...")
input()
print("Continuing...")

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 3  # give a unique id to each object we interact with (it can be any integers) - for second object

# Let's now move on to the second object we want to track (giving it object id `3`)
# with a positive click at (x, y) = (400, 150)
points = np.array([[1048, 527]], dtype=np.float32) # points identifying second object
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32) # label identifying second object
prompts[ann_obj_id] = points, labels

# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)
# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.show()

print("Press enter to continue...")
input()
print("Continuing...")

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }


# define the figure outside of the loop
fig = plt.figure(figsize=(12,8))
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.title(f"frame {out_frame_idx}")
    im=plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])),animated=True)
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask,plt.gca(),obj_id=out_obj_id)

        # here save the files with an increasing index in the folder called output
    plt.savefig(f'output/s_multi{out_frame_idx}.png')
    # plt.show()

# run ffmpeg -framerate 30 -i s_multi%d.png -c:v lib264 -r 30 output_multi.mp4