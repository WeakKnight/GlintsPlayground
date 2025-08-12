import bsdf;
import glints2023;
import utils;

struct Uniforms
{
    float2 screenSize;
    float focalLength, frameHeight;
    float3 cameraDir;
    float3 cameraUp;
    float3 cameraRight;
    float3 cameraPosition;
};

float4 raytraceScene(
    int2 threadIdx,
    Uniforms uniforms,
    Texture2D albedoTex,
    Texture2D normalTex,
    Texture2D roughnessTex,
    Texture2D<float4> noiseTex,
    SamplerState samplerState)
{
    if (threadIdx.x >= (int)uniforms.screenSize.x) return 0;
    if (threadIdx.y >= (int)uniforms.screenSize.y) return 0;

    float frameWidth = uniforms.screenSize.x / uniforms.screenSize.y * uniforms.frameHeight;
    float imageY = (threadIdx.y / uniforms.screenSize.y - 0.5f) * uniforms.frameHeight;
    float imageX = (threadIdx.x / uniforms.screenSize.x - 0.5f) * frameWidth;

    float imageYUp = ((threadIdx.y + 1) / uniforms.screenSize.y - 0.5f) * uniforms.frameHeight;
    float imageXRight = ((threadIdx.x + 1) / uniforms.screenSize.x - 0.5f) * frameWidth;
    
    float imageZ = uniforms.focalLength;

    float3 rayDir = normalize(uniforms.cameraDir.xyz * imageZ - uniforms.cameraUp.xyz * imageY + uniforms.cameraRight.xyz * imageX);

    float3 resultColor = 0;
    float t = 0;
    float u = 0, v = 0;

    if (raySquareIntersection(uniforms.cameraPosition, rayDir, float3(0, 0, 0), float3(0, 1, 0), float3(1, 0, 0), 40.0f, t, u, v))
    {
        float2 uv = float2(u, v);
        float2 duvdx;
        float2 duvdy;
        {
            float tt;
            float uu;
            float vv;
            float3 rayDirRight = normalize(uniforms.cameraDir.xyz * imageZ - uniforms.cameraUp.xyz * imageY + uniforms.cameraRight.xyz * imageXRight);
            float3 rayDirUp = normalize(uniforms.cameraDir.xyz * imageZ - uniforms.cameraUp.xyz * imageYUp + uniforms.cameraRight.xyz * imageX);

            raySquareIntersection(uniforms.cameraPosition, rayDirRight, float3(0, 0, 0), float3(0, 1, 0), float3(1, 0, 0), 40.0f, tt, uu, vv);
            duvdx = float2((uu - u), (vv - v));

            raySquareIntersection(uniforms.cameraPosition, rayDirUp, float3(0, 0, 0), float3(0, 1, 0), float3(1, 0, 0), 40.0f, tt, uu, vv);
            duvdy = float2((uu - u), (vv - v));
        }

        float tillingFactor = 6;
        float2 tilledTexcoord = float2(u, v);
        tilledTexcoord *= tillingFactor;
        float2 tilled_duvdx = duvdx * tillingFactor;
        float2 tilled_duvdy = duvdy * tillingFactor;

        float3 Le = 15;
        float3 V = -rayDir;
        float3 N = normalize(textureNoTile(normalTex, samplerState, tilledTexcoord) * 2 - 1).xzy;
        float3 L = normalize(float3(0.6, .9, 0.5));
        float3 baseColor = textureNoTile(albedoTex, samplerState, tilledTexcoord);     // albedoTex.SampleLevel(samplerState, texcoord, 0).rgb;
        float roughness = textureNoTile(roughnessTex, samplerState, tilledTexcoord).r; // roughnessTex.SampleLevel(samplerState, texcoord, 0).r;
        float alpha = roughness * roughness;
        float metallic = 0.25;
        float3 diffuse = baseColor * (1.0 - metallic);
        diffuse = clamp(diffuse, 0.01, 0.95);
        float3 specular = lerp(0.04, baseColor, metallic);

        Frame shadingFrame = Frame(N);
        float3 wi = shadingFrame.toLocal(V);
        float3 wo = shadingFrame.toLocal(L);

        float3 diffuseLobe = diffuse * M_1_PI;
        float3 specularLobe;
        if (min(wi.z, wo.z) < kMinCosTheta)
        {
            specularLobe = 0;
        }
        else
        { 
            float3 h = normalize(wi + wo);
            float wiDotH = dot(wi, h);

            float D = evalNdfGGX(alpha, h.z);
            if (threadIdx.x > ((int)uniforms.screenSize.x / 2))
            {
                float maxNDF = 1.0 / (M_PI * alpha * alpha);
                D = SampleGlints2023NDF(noiseTex, h, D, maxNDF, tilledTexcoord, tilled_duvdx, tilled_duvdy);
            }

            float G = evalMaskingSmithGGXCorrelated(alpha, wi.z, wo.z);
            float3 F = evalFresnelSchlick(specular, 1.f, wiDotH);
            specularLobe = F * D * G * 0.25f / wi.z;
        }

        // diffuseLobe = 0;
        resultColor = Le * saturate(dot(N, L)) * (diffuseLobe + specularLobe);
        // resultColor = float3(frac(tilledTexcoord), 0);
        // resultColor = float3(frac(tilled_duvdy), 0);
    }

    resultColor = resultColor / (1 + resultColor);

    if (abs(threadIdx.x - ((int)uniforms.screenSize.x / 2)) < 2)
    {
        resultColor = 0;
    }

    return float4(resultColor, 1);
}