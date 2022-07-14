import os
import argparse

import torch
import torch.nn as nn
from model import network
from data import dataset


def train(conf):
	if not os.path.exists(conf.model_path):
		os.mkdir(conf.model_path)

	train_loader, num_class = dataset.get_train_loader(conf.dataset,
	                                                   conf.dataset_path,
	                                                   conf.img_size,
	                                                   conf.batch_size)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	cnn = network.CNN(img_size=conf.img_size, num_class=num_class).to(device)
	criterion = nn.CrossEntropyLoss().to(device)

	optimizer = torch.optim.Adam(cnn.parameters(), lr=conf.lr)

	min_loss = 999

	print("[INFO] Starting training...")
	for epoch in range(conf.epoch):
		epoch_loss = 0
		for i, (images, labels) in enumerate(train_loader):
			images, labels = images.to(device), labels.to(device)
			optimizer.zero_grad()
			outputs, _ = cnn(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			epoch_loss += loss.item()
			if (i + 1) % conf.log_step == 0:
				if conf.save_model_in_epoch:
					torch.save(cnn.state_dict(), os.path.join(
						conf.model_path, conf.model_name))
				print(f'Epoch [{epoch + 1}/{conf.epoch}],'
				      f' Iter [{i + 1}/{len(train_loader)}], Loss: '
				      f'{loss.item():.4f}')

		avg_epoch_loss = epoch_loss / len(train_loader)
		print(f'Epoch [{epoch + 1}/{conf.epoch}],'
		      f' Loss: {avg_epoch_loss:.4f}')
		if avg_epoch_loss < min_loss:
			min_loss = avg_epoch_loss
			torch.save(cnn.state_dict(),
			           os.path.join(conf.model_path, conf.model_name))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='CIFAR',
	                    choices=['STL', 'CIFAR', 'OWN'])
	parser.add_argument('--dataset_path', type=str, default='./data')
	parser.add_argument('--model_path', type=str, default='./model')
	parser.add_argument('--model_name', type=str, default='model.pth')

	parser.add_argument('--img_size', type=int, default=128)
	parser.add_argument('--batch_size', type=int, default=32)

	parser.add_argument('--epoch', type=int, default=30)
	parser.add_argument('--log_step', type=int, default=10)
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('-s', '--save_model_in_epoch', action='store_true')
	config = parser.parse_args()
	print(config)

	train(config)
