import glob 
import os 
import sys 

def run_model(audio_path, video_path, save_dir, use_gan=False):
    if use_gan:
        ckpt_path = 'checkpoints/wav2lip_gan.pth'
    else:
        ckpt_path = 'checkpoints/wav2lip.pth'
    video_name = 'result_voice'
    cmd = f"python inference.py --checkpoint_path {ckpt_path} --face {video_path} --audio {audio_path} --outfile {save_dir}/{video_name}.mp4 "
    print(cmd)
    os.system(cmd) 



if __name__ == '__main__':
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

    talking_head_path_fmt = '/home/users/abhishekm/art-flow/fomm-relative-videos-2-images-1/{video_name}/{image_name}/result.mp4'
    polly_audio_file = '/home/users/abhishekm/art-flow/golden-set/audio_polly/Polly/v{version_index}/{audio_index}_v{version_index}.mp3'
    deepzen_audio_file = '/home/users/abhishekm/art-flow/golden-set/audio_deepzen/DeepZen.io/v{version_index}/{audio_index:02d}_v{version_index}.wav'
    talking_head_paths = glob.glob(talking_head_path_fmt) 
    save_dir = '/home/users/abhishekm/art-flow/fomm-relative-wav2lip-v3-deepzen/'
    for video in videos:
        for image in images:
            talking_head_path = talking_head_path_fmt.format(video_name=video, image_name=image)
            driving_video_name = video
            source_image_name = image
            save_dir_talking_head = os.path.join(save_dir, driving_video_name, source_image_name) 
            for version_index in [1]:
                for audio_index in [1, 5, 8]:
                    # audio = polly_audio_file.format(version_index=version_index, audio_index=audio_index)
                    audio = deepzen_audio_file.format(version_index=version_index, audio_index=audio_index)
                    run_model(audio, talking_head_path, save_dir_talking_head, use_gan=False)





