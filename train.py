import osimport argparseimport jsonimport torchimport torch.nn as nnimport torch.nn.functional as Ffrom torch.cuda.amp import autocast, GradScalerfrom torch.utils.data import DataLoaderfrom tensorboardX import SummaryWriterfrom tqdm import tqdmfrom utils import AverageMeterfrom datasets.loader import PairLoaderfrom models import *import wandbimport sysimport datetimeimport torch.fft as fftisDebug = True if sys.gettrace() else Falseparser = argparse.ArgumentParser()parser.add_argument('--model', default='dehazeformer-s', type=str, help='model name')parser.add_argument('--num_workers', default=16, type=int, help='number of workers')parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')# ../../Uformer-main/data/train/parser.add_argument('--data_dir', default='../../Uformer-main/data/', type=str, help='path to dataset')parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')parser.add_argument('--exp', default='endoscopic', type=str, help='experiment setting')parser.add_argument('--gpu', default='0,1', type=str, help='GPUs used for training')args = parser.parse_args()if isDebug:	args.num_workers = 0	os.system('wandb disabled')else:	os.system('wandb enabled')os.environ['CUDA_VISIBLE_DEVICES'] = args.gpudef train(train_loader, network, criterion, optimizer, scaler):	losses = AverageMeter()	torch.cuda.empty_cache()		network.train()	use_freq_loss = True	for batch in train_loader:		source_img = batch['source'].cuda()		target_img = batch['target'].cuda()		with autocast(args.no_autocast):			output = network(source_img)			loss = criterion(output, target_img)			if use_freq_loss:				output = output * 0.5 + 0.5				target_img = target_img * 0.5 + 0.5				output_fft = fft.fftn(torch.tensor(output), dim=(2, 3))				target_fft = fft.fftn(torch.tensor(target_img), dim=(2, 3))				freq_loss = criterion(output_fft, target_fft)				wandb.log({"freq_loss": freq_loss.item()})				loss += freq_loss		losses.update(loss.item())		wandb.log({"loss": loss.item()})		optimizer.zero_grad()		scaler.scale(loss).backward()		scaler.step(optimizer)		scaler.update()	return losses.avgdef valid(val_loader, network):	PSNR = AverageMeter()	torch.cuda.empty_cache()	network.eval()	for batch in val_loader:		source_img = batch['source'].cuda()		target_img = batch['target'].cuda()		with torch.no_grad():							# torch.no_grad() may cause warning			output = network(source_img).clamp_(-1, 1)				mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))		psnr = 10 * torch.log10(1 / mse_loss).mean()		PSNR.update(psnr.item(), source_img.size(0))	return PSNR.avgif __name__ == '__main__':	setting_filename = os.path.join('configs', args.exp, args.model+'.json')	if not os.path.exists(setting_filename):		setting_filename = os.path.join('configs', args.exp, 'default.json')	with open(setting_filename, 'r') as f:		setting = json.load(f)	wandb.init(project="DehazeFormer", entity="wuhx")	wandb.config = setting	network = eval(args.model.replace('-', '_'))()	if isDebug:		from ptflops import get_model_complexity_info		inp_shape = (3, 256, 256)		macs, params = get_model_complexity_info(network, inp_shape, verbose=False, print_per_layer_stat=False)		with open('info.txt','w') as f:			f.write(f'macs: {macs}  paramsL {params}')	network = nn.DataParallel(network).cuda()	criterion = nn.L1Loss()	if setting['optimizer'] == 'adam':		optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])	elif setting['optimizer'] == 'adamw':		optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])	else:		raise Exception("ERROR: unsupported optimizer") 	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'], eta_min=setting['lr'] * 1e-2)	scaler = GradScaler()	dataset_dir = args.data_dir	train_dataset = PairLoader(dataset_dir, 'train', 'train', 								setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])	train_loader = DataLoader(train_dataset,                              batch_size=setting['batch_size'],                              shuffle=True,                              num_workers=args.num_workers,                              pin_memory=True,                              drop_last=True)	val_dataset = PairLoader(dataset_dir, 'val', setting['valid_mode'],							  setting['patch_size'])	val_loader = DataLoader(val_dataset,                            batch_size=setting['batch_size'],                            num_workers=args.num_workers,                            pin_memory=True)	today = str(datetime.date.today())	save_dir = os.path.join(args.save_dir+args.exp, today)	os.makedirs(save_dir, exist_ok=True)	if not os.path.exists(os.path.join(save_dir, args.model+'.pth')):		print('==> Start training, current model name: ' + args.model)		# print(network)		writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))		best_psnr = 0		for epoch in tqdm(range(setting['epochs'] + 1)):			loss = train(train_loader, network, criterion, optimizer, scaler)			writer.add_scalar('train_loss', loss, epoch)			scheduler.step()			if epoch % setting['eval_freq'] == 0:				avg_psnr = valid(val_loader, network)				wandb.log({"valid_psnr": avg_psnr})				writer.add_scalar('valid_psnr', avg_psnr, epoch)				if avg_psnr > best_psnr:					best_psnr = avg_psnr					torch.save({'state_dict': network.state_dict()},                			   os.path.join(save_dir, args.model+'.pth'))								writer.add_scalar('best_psnr', best_psnr, epoch)			if epoch % 5 == 0:				torch.save({'state_dict': network.state_dict()},						   os.path.join(save_dir, args.model +str(epoch)+ '.pth'))	else:		print('==> Existing trained model')		exit(1)