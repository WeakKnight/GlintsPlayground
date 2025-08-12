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

float2 getOffset(float2 p) {
    return float2(frac(sin(p.x * 946336. + 6. + p.y * 334747.)),
                  frac(sin(p.x * 756854. + p.y * 95236. + 1.)));
}

#define HASHSCALE3 float3(.1031, .1030, .0973)
float2 noise(float2 p) {
    float3 p3 = frac(float3(p.xyx) * HASHSCALE3);
    p3 += dot(p3, p3.yzx + 19.19);
    return frac(float2((p3.x + p3.y) * p3.z, (p3.x + p3.z) * p3.y));
}

float2 random_uv(float2 tuv, float offset) {
    float2 noise_of = noise(floor(tuv * float2(1.0 / 3.0, 1.0 / 2.0)) + offset);
    return (tuv + noise_of) * (sign(fmod(noise_of, 0.001) - 0.0005));
}

float3 textureNoTile(Texture2D texture, SamplerState samplerState, float2 uv)
{
#if 0
    float overlap = 0.1;

    float w = 0.;
    float3 col = float3(0.);
    float2 uvf = fract(uv);
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            float2 off = float2(i, j);
            float2 uv2 = off + float2((i == 0) ? uvf.x : 1. - uvf.x, (j == 0) ? uvf.y : 1. - uvf.y);

            float weight = (1. - smoothstep(0., overlap, abs(uvf.x - uv2.x)))
                          * (1. - smoothstep(0., overlap, abs(uvf.y - uv2.y)));

            w += weight;
            col += weight * texture.SampleLevel(samplerState, uvf + getOffset(floor(uv) + off), 0).rgb;
        }
    }
    col /= w;
    return col;
#else
    float2 new_uv = uv;
    float con1 = 0.25;
    float con2 = 0.25;

    float2 new_tuv0 = uv + (fmod(new_uv.x, 3.0) < 1.5 ? float2(0.0, 0.0) : float2(0.0, 1.0));
    float2 new_tuv1 = uv + (fmod(new_uv.x + 0.75, 3.0) < 1.5 ? float2(0.25, 1.0) : float2(0.25, 0.0));
    float2 new_tuv2 = uv + (fmod(new_uv.x + 1., 3.0) < 1.5 ? float2(1.0, 0.0) : float2(0.5, 1.0));

    float2 random_tuv0 = random_uv(new_tuv0, 0.0);
    float2 random_tuv1 = random_uv(new_tuv1, 1.0);
    float2 random_tuv2 = random_uv(new_tuv2, 2.0);

    float4 white = texture.SampleLevel(samplerState, random_tuv0, 0);
    float4 black = texture.SampleLevel(samplerState, random_tuv1, 0);
    float4 red = texture.SampleLevel(samplerState, random_tuv2, 0);

    float4 mix_fact = float4(uv, uv);
    mix_fact.z += 1.0;

    mix_fact.yw += step(1.5, fmod(mix_fact.xz, 3.0));
    mix_fact.yw = abs(fmod(mix_fact.yw, 2.0) - 1.0);
    mix_fact.xz = abs(fmod(mix_fact.xz, 1.5) - 0.75);

    mix_fact.xz = smoothstep(0.5, 0.5 + con1, mix_fact.xz);
    mix_fact.yw = smoothstep(0.5, 0.5 + con2, mix_fact.yw);

    float2 mix_fact_12 = mix_fact.xz + mix_fact.yw - mix_fact.xz * mix_fact.yw;
    float4 unrep = lerp(white, black, mix_fact_12.x);
    unrep = lerp(red, unrep, mix_fact_12.y);
    return unrep.rgb;
#endif
}

// ray differentials
void calcDpDxy(in float3 ro, in float3 rd, in float3 rdx, in float3 rdy, in float t, in float3 nor,
               out float3 dpdx, out float3 dpdy) 
{
    dpdx = 2. * t * (rdx * dot(rd, nor) / dot(rdx, nor) - rd) * sign(dot(rd, rdx));
    dpdy = 2. * t * (rdy * dot(rd, nor) / dot(rdy, nor) - rd) * sign(dot(rd, rdy));
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

    if (raySquareIntersection(uniforms.cameraPosition, rayDir, float3(0, 0, 0), float3(0, 1, 0), float3(1, 0, 0), 40.0f, t, u, v))
    {
        float2 texcoord = float2(u, v);
        texcoord = texcoord * 6;

        float3 Le = 9;
        float3 V = -rayDir;
        float3 N = normalize(textureNoTile(normalTex, samplerState, texcoord) * 2 - 1).xzy;
        float3 L = normalize(float3(0.6, .9, 0.5));
        float3 baseColor = textureNoTile(albedoTex, samplerState, texcoord); // albedoTex.SampleLevel(samplerState, texcoord, 0).rgb;
        float roughness = textureNoTile(roughnessTex, samplerState, texcoord).r; // roughnessTex.SampleLevel(samplerState, texcoord, 0).r;
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