import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from torchvision import utils as vutils
import time


def main(config):
    # logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        # batch_size=6,
        batch_size=1,
        shuffle=False,
        validation_split=0.0,
        # num_workers=2,
        num_workers=0,
        view_dirname=config['data_loader']['args']['view_dirname'],
        depth_dirname=config['data_loader']['args']['depth_dirname'],
        flow_dirname=config['data_loader']['args']['flow_dirname'],
        downscale_factor=config['data_loader']['args']['downscale_factor']
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    # logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (img_view, img_depth, img_flow, img_view_truth) in enumerate(tqdm(data_loader)):
            # img_view = img_view.to(self.device)
            # img_depth = img_depth.to(self.device)
            # img_flow = img_flow.to(self.device)
            # img_view_truth = img_view_truth.to(self.device)
            img_view = img_view.to(device)
            img_depth = img_depth.to(device)
            img_flow = img_flow.to(device)
            img_view_truth = img_view_truth.to(device)

#             print(img_view.shape,
#             img_depth.shape, 
#             img_flow.shape, 
# img_view_truth.shape
#             )

            target = img_view_truth[:,:,0,:,:]
            vutils.save_image(target[0], './output_test/target.png'.format(i))

            output = model(img_view , img_depth , img_flow)
            # print("upsampling time = ", end-start, 's')
            vutils.save_image(output, './output_test/output_{}.png'.format(i))

            # TODO remove
            break

            # #
            # # save sample images, or do something with output here
            # #

            # # computing loss, metrics on test set
            # loss = loss_fn(output, target, 0.1)
            # batch_size = img_view.shape[0]
            # total_loss += loss.item() * batch_size
            # for i, metric in enumerate(metric_fns):
            #     total_metrics[i] += metric(output, target) * batch_size

    # n_samples = len(data_loader.sampler)
    # log = {'loss': total_loss / n_samples}
    # log.update({
    #     met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    # })
    # logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
