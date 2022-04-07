import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np
import os 
import glob
from joblib import Parallel, delayed
import subprocess
from tqdm import tqdm 
import cv2 

def stitch_frames(frames_dir, audio_name, total_frames):
	audio_version = audio_name.split('_')[1]
	video_path = os.path.splitext(frames_dir)[0] + '.mp4'
	audio_path = f'/home/users/abhishekm/art-flow/golden-set/audio_polly/Polly/{audio_version}/{audio_name}.mp3'
	# ffmpeg_command = f"/home/users/abhishekm/temp/ffmpeg-5.0-amd64-static/ffmpeg  -framerate 25 -i '{frames_dir}/%6d.jpg' -i {audio_path} {video_path}"
	if not audio_path.endswith('.wav'):
		print('Extracting raw audio...')
		command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_path, 'temp/temp.wav')

		subprocess.call(command, shell=True)
		audio_path = 'temp/temp.wav'
	frame = cv2.imread(f'{frames_dir}/{0:06d}.jpg')
	frame = np.asarray(frame)
	h, w, _ = frame.shape 
	out = cv2.VideoWriter('temp/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), 25, (w, h))
	for idx in tqdm(range(total_frames)):
		frame = cv2.imread(f'{frames_dir}/{idx:06d}.jpg')
		out.write(frame) 
	out.release()

	ffmpeg_command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_path, 'temp/result.avi', video_path)
	print(ffmpeg_command)
	os.system(ffmpeg_command)

def plot_lipsync_images(audio_name, video_name, root_dir, total_frames):
	images = ['Test_06',
			'image_10',
			'source_image_6',
			'image_8',
			'image_14',
			'image',
			'Test_03',
			'source_image_3',
			'Test_02',
			'image_11',
			'image_9',
			'source_image_4',
			'image_7',
			'Test_07']
	save_dir = os.path.join(root_dir, video_name, f'lipsync_mosaic_{audio_name}')
	os.makedirs(save_dir, exist_ok=True) 
	layout_dict = np.arange(15).reshape((5, 3))
	for idx in range(total_frames):
		fig = plt.figure(figsize=(5, 3), clear=True)
		ax_dict = fig.subplot_mosaic(
			layout_dict
		)
		plt.subplots_adjust(wspace=-0.5, hspace=0.1)
		for img_idx, image in enumerate(images):
			frame_path = os.path.join(root_dir, video_name, image, audio_name, 'lipsync',f'{idx:06d}.jpg')
			im = Image.open(frame_path) 
			ax_dict[img_idx].imshow(im) 
		
		for k in ax_dict:
			ax_dict[k].axis('off') 
		fig.tight_layout() 
		fig.savefig(os.path.join(save_dir, f'{idx:06d}.jpg'))
		plt.close(fig)
	stitch_frames(save_dir, audio_name, total_frames)
	

if __name__  == '__main__':
	images = ['Test_06',
	'image_10',
	'source_image_6',
	'image_8',
	'image_14',
	'image',
	'Test_03',
	'source_image_3',
	'Test_02',
	'image_11',
	'image_9',
	'source_image_4',
	'image_7',
	'Test_07']

	videos = ['drive2_vid08',
			'drive2_vid09']

	# audio_names = ['10_v1', '1_v1', '5_v1'] 
	audio_names = ['1_v1']
	root_dir = '/home/users/abhishekm/art-flow/fomm-relative-wav2lip-v2/'
	params = []
	for audio_name in audio_names:
		for video in videos:
			lipsync_frames = os.path.join(root_dir, video, images[0], audio_name, 'lipsync','*.jpg')
			frames = glob.glob(lipsync_frames)
			total_frames = len(frames)
			param = {
				'audio_name':audio_name, 
				'video_name':video, 
				'root_dir':root_dir, 
				'total_frames':total_frames
			}
			params.append(param)
			# print(param)
			# plot_lipsync_images(audio_name, videos[0], root_dir, total_frames)
	
	Parallel(n_jobs=10)(delayed(plot_lipsync_images)(**param) for param in params)
	
	
