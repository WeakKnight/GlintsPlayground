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
module = spy.Module.load_from_file(device, "glints2016.slang")

uniforms = {
    "_type": "Uniforms",
    "screenSize": sgl.float2(512, 512),
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
imageData = imageio.v3.imread("imageio:bricks.jpg")

# reshape imageData into 512x512x4, adding an alpha channel
if imageData.shape[2] == 3:
    imageData = np.concatenate([imageData, np.ones((512, 512, 1), dtype=np.uint8) * 255], axis=2)

tex = device.create_texture(
    width = 512,
    height = 512,
    format = sgl.Format.rgba8_unorm,
    usage = sgl.ResourceUsage.shader_resource | sgl.ResourceUsage.unordered_access,
    data = imageData,
)
samplerState = device.create_sampler()

output = device.create_texture(
    width = 512,
    height = 512,
    format = sgl.Format.rgba8_unorm,
    usage = sgl.ResourceUsage.shader_resource | sgl.ResourceUsage.unordered_access
)
module.raytraceScene(call_id(), uniforms, tex, samplerState, _result = output)

imageio.imwrite("out.png", output.to_numpy())