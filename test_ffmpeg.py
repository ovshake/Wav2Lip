import os 
import glob 
import os.path as osp 
import cv2 
import subprocess

if __name__ == '__main__':
	audio_path = '/home/users/abhishekm/art-flow/golden-set/audio_polly/Polly/v1/10_v1.mp3'
	frames_dir = '/home/users/abhishekm/art-flow/fomm-relative-wav2lip-v3/drive2_vid08/image_10/10_v1/lipsync/'
	frames_fmt = osp.join(frames_dir, '*.jpg')
	all_frames = glob.glob(frames_fmt)
	total_frames = len(all_frames) 
	frame_w, frame_h = 256, 256
	fps = 30
	out = cv2.VideoWriter('temp/result.avi', 
									cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

	save_dir = 'test-ffmpeg'
	if not audio_path.endswith('.wav'):
		print('Extracting raw audio...')
		command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_path, 'temp/temp.wav')

		subprocess.call(command, shell=True)
		audio_path = 'temp/temp.wav'
	
	for i in range(total_frames):
		frame_filename = "{idx:06d}.jpg"
		frame_path = osp.join(frames_dir, frame_filename.format(idx=i)) 
		f = cv2.imread(frame_path) 
		out.write(f) 
	out.release()

	os.makedirs(save_dir, exist_ok=True)
	command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_path, 'temp/result.avi', f'{save_dir}/video-2.mp4')

	subprocess.call(command, shell=True)


	




