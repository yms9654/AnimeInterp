
testset_root = '/mnt/c/data/0415/image'
test_flow_root = '/mnt/c/data/0415/flow'

test_size = (960, 540)
test_crop_size = (960, 540)

mean = [0., 0., 0.]
std  = [1, 1, 1]

# inter_frames = 1
inter_frames = 5

model = 'AnimeInterp'
pwc_path = None

checkpoint = 'checkpoints/anime_interp_full.ckpt'

store_path = 'outputs/avi_full_results'