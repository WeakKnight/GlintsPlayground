# SPDX-License-Identifier: Apache-2.0

import slangpy as spy
import sgl
from pathlib import Path
import numpy as np
from slangpy.types import call_id
import math
import imageio
import os

device = spy.create_device(include_paths = [Path(__file__).parent])
module = spy.Module.load_from_file(device, "glints.slang")

output_w = 1920
output_h = 1080

uniforms = {
    "_type": "Uniforms",
    "screenSize": sgl.float2(output_w, output_h),
    "focalLength": 24.0,
    "frameHeight": 24.0,
}

radius = 15.0
alpha: float = 0.0
pi: float = 3.14159265359
beta: float = -pi / 4.0

def updateCamera():
    global alpha, beta, pi
    cameraDir = [
        -math.cos(alpha) * math.sin(beta),
        -math.cos(beta),
        -math.sin(alpha) * math.sin(beta),
    ]
    betaUp = beta + pi * 0.5
    cameraUp = [
        math.cos(alpha) * math.sin(betaUp),
        math.cos(betaUp),
        math.sin(alpha) * math.sin(betaUp),
    ]
    cameraRight = np.cross(cameraDir, cameraUp).tolist()
    cameraPos = [-cameraDir[0] * radius, -cameraDir[1] * radius, -cameraDir[2] * radius]
    uniforms["cameraDir"] = cameraDir
    uniforms["cameraUp"] = cameraUp
    uniforms["cameraRight"] = cameraRight
    uniforms["cameraPosition"] = cameraPos
    return

updateCamera()

pathname = os.path.realpath(__file__)

def loadImageData(path, w, h):
    imageData = imageio.v3.imread(path)
    # reshape imageData into 512x512x4, adding an alpha channel
    print(imageData.shape)
    if len(imageData.shape) >= 3 and imageData.shape[2] == 3:
        imageData = np.concatenate([imageData, np.ones((w, h, 1), dtype=np.uint8) * 255], axis=2)
    return imageData

baseColorImageData = loadImageData("albedo.jpg", 4096, 4096)
normalImageData = loadImageData("normal.jpg", 4096, 4096)
roughnessImageData = loadImageData("roughness.jpg", 4096, 4096)

albedoTex = device.create_texture(
    width = 4096,
    height = 4096,
    format = sgl.Format.rgba8_unorm_srgb,
    usage = sgl.ResourceUsage.shader_resource,
    data = baseColorImageData,
)

normalTex = device.create_texture(
    width = 4096,
    height = 4096,
    format = sgl.Format.rgba8_unorm,
    usage = sgl.ResourceUsage.shader_resource,
    data = normalImageData,
)

roughnessTex = device.create_texture(
    width = 4096,
    height = 4096,
    format = sgl.Format.r8_unorm,
    usage = sgl.ResourceUsage.shader_resource,
    data = roughnessImageData,
)

samplerState = device.create_sampler()

output = device.create_texture(
    width = output_w,
    height = output_h,
    format = sgl.Format.rgba8_unorm,
    usage = sgl.ResourceUsage.shader_resource | sgl.ResourceUsage.unordered_access
)
module.raytraceScene(call_id(), uniforms, albedoTex, normalTex, roughnessTex, samplerState, _result = output)

imageio.imwrite("out.png", output.to_numpy())