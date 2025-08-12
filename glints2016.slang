// SPDX-License-Identifier: Apache-2.0
import bsdf;

struct Uniforms
{
    float2 screenSize;
    float focalLength, frameHeight;
    float3 cameraDir;
    float3 cameraUp;
    float3 cameraRight;
    float3 cameraPosition;
};

bool rayPlaneIntersection(float3 rayOrigin, float3 rayDir, float3 normal, float3 center, out float t)
{
    float denom = dot(normal, rayDir);
    if (abs(denom) > 0.0001f)
    {
        t = dot(center - rayOrigin, normal) / denom;
        return t >= 0;
    }
    t = 0;
    return false;
}

bool raySquareIntersection(float3 rayOrigin, float3 rayDir, float3 center, float3 normal, float3 right, float halfEdgeLength, out float t, out float u, out float v)
{
    u = 0;
    v = 0;
    if (rayPlaneIntersection(rayOrigin, rayDir, normal, center, t))
    {
        float3 intersectionPoint = rayOrigin + rayDir * t;
        float3 toCenter = intersectionPoint - center;
        float3 up = cross(normal, right);
        float rightLength = dot(right, toCenter);
        float upLength = dot(up, toCenter);
        u = rightLength / halfEdgeLength * 0.5 + 0.5;
        v = upLength / halfEdgeLength * 0.5 + 0.5;
        return abs(rightLength) <= halfEdgeLength && abs(upLength) <= halfEdgeLength;
    }
    return false;
}

float3 evalSpecular(float3 wi, float3 wo, float alpha, float3 albedo)
{
    if (min(wi.z, wo.z) < kMinCosTheta)
        return float3(0.f);
    float3 h = normalize(wi + wo);
    float wiDotH = dot(wi, h);

    float D = evalNdfGGX(alpha, h.z);
    float G = evalMaskingSmithGGXCorrelated(alpha, wi.z, wo.z);
    float3 F = evalFresnelSchlick(albedo, 1.f, wiDotH);
    return F * D * G * 0.25f / wi.z;
}

float4 raytraceScene(
    int2 threadIdx,
    Uniforms uniforms,
    Texture2D albedoTex,
    Texture2D normalTex,
    Texture2D roughnessTex,
    SamplerState samplerState)
{
    if (threadIdx.x >= (int)uniforms.screenSize.x) return 0;
    if (threadIdx.y >= (int)uniforms.screenSize.y) return 0;

    float frameWidth = uniforms.screenSize.x / uniforms.screenSize.y * uniforms.frameHeight;
    float imageY = (threadIdx.y / uniforms.screenSize.y - 0.5f) * uniforms.frameHeight;
    float imageX = (threadIdx.x / uniforms.screenSize.x - 0.5f) * frameWidth;
    float imageZ = uniforms.focalLength;
    float3 rayDir = normalize(uniforms.cameraDir.xyz * imageZ - uniforms.cameraUp.xyz * imageY + uniforms.cameraRight.xyz * imageX);

    float3 resultColor = 0;
    float t = 0;
    float u = 0, v = 0;

    if (raySquareIntersection(uniforms.cameraPosition, rayDir, float3(0, 0, 0), float3(0, 1, 0), float3(1, 0, 0), 5.0f, t, u, v))
    {
        float2 texcoord = float2(u, v);

        float3 Le = 9;
        float3 V = -rayDir;
        float3 N = normalize(normalTex.SampleLevel(samplerState, texcoord, 0).rgb * 2 - 1).xzy;
        float3 L = normalize(float3(0.6, .9, 0.5));
        float3 baseColor = albedoTex.SampleLevel(samplerState, texcoord, 0).rgb;
        float roughness = roughnessTex.SampleLevel(samplerState, texcoord, 0).r;
        float alpha = roughness * roughness;
        float metallic = 0;
        float3 diffuse = baseColor * (1.0 - metallic);
        diffuse = clamp(diffuse, 0.01, 0.95);
        float3 specular = lerp(0.04, baseColor, metallic);

        Frame shadingFrame = Frame(N);
        float3 wi = shadingFrame.toLocal(V);
        float3 wo = shadingFrame.toLocal(L);

        float3 diffuseLobe = diffuse * M_1_PI;
        float3 specularLobe = evalSpecular(wi, wo, alpha, specular);
        resultColor = Le * saturate(dot(N, L)) * (diffuseLobe + specularLobe);
    }

    resultColor = resultColor / (1 + resultColor);

    return float4(resultColor, 1);
}