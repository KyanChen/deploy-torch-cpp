import cv2
import tifffile
import torch
from torch import nn
import onnx
import onnxruntime
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x5 = x5 * 0.2 + x
        # Emperically, we use 0.2 to scale the residual for better performance
        return x5


class RRDB(nn.Module):

    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = out * 0.2 + x
        # Emperically, we use 0.2 to scale the residual for better performance
        return out


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        num_in_ch = num_in_ch * 4
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        b, c, h, w = x.shape
        x_view = x.view(b, c, h//2, 2, w//2, 2)
        feat = x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, -1, h//2, w//2)
        feat = self.conv_first(feat)
        feat1 = self.body(feat)
        feat1 = self.conv_body(feat1)
        feat = feat + feat1
        feat = self.lrelu(self.conv_up1(self.up_sample(feat)))
        feat = self.lrelu(self.conv_up2(self.up_sample(feat)))
        feat = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return feat


if __name__ == '__main__':
    # create resnet18 model from torchvision
    # model = torchvision.models.resnet18(pretrained=True)
    model_path = 'ckpt/net_g_DF2K_RRDB+LDL_x2.pth'
    img_file = 'sample.tiff'
    crop_size = 512
    model = RRDBNet()
    sd = torch.load(model_path)
    model.load_state_dict(sd, strict=True)
    model = model.to(device)
    model.eval()

    img = tifffile.imread(img_file)
    img = img / (255.0 * 8)
    img = torch.from_numpy(img).float().to(device)
    img = img.unsqueeze(0).unsqueeze(0)
    b, c, h, w = img.shape
    img = img.view(b, c, h//crop_size, crop_size, w//crop_size, crop_size)
    img = img.permute(0, 2, 4, 1, 3, 5).reshape(-1, 1, crop_size, crop_size)
    dst_tensor = img.repeat(1, 3, 1, 1)
    dst_tensor = dst_tensor.to(device)
    with torch.no_grad():
        output_ori = model(dst_tensor)

    # export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
    # export_output = torch.onnx.dynamo_export(model, dst_tensor, export_options)
    # export_output.save("rrdbnet.onnx")
    # export onnx with dynamic batch size

    torch.onnx.export(
        model, dst_tensor,
        "rrdbnet.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        verbose=True,
    )

    ort_session = onnxruntime.InferenceSession("rrdbnet.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: dst_tensor.cpu().numpy()}
    output_script = ort_session.run(None, ort_inputs)[0]
    output_script = torch.from_numpy(np.array(output_script)).to(device)

    cv2.imwrite('output_ori.png', output_script[0, 0].cpu().numpy() * 255)

    # check the output
    print("output_ori: ", output_ori.shape)
    print("output_script: ", output_script.shape)
    print("output_ori - output_script: ", torch.sum(output_ori - output_script))



