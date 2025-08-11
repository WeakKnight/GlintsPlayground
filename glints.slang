// Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
// Created by Tobias Zirr (KIT, NVIDIA) and Anton Kaplanyan (NVIDIA)
// License Creative Commons Attribution 3.0 Unported License.

// For a working self-contained example, find shadertoy at:
// https://www.shadertoy.com/view/ldVGRh

// Example code for the paper "Real-time Rendering of Procedural Multiscale Materials",
// Tobias Zirr (NVIDIA / KIT), Anton Kaplanyan (NVIDIA),
// in ACM SIGGRAPH Symposium on Interactive 3D Graphics and Games, February 2016.
// More info on https://research.nvidia.com/publication/real-time-rendering-procedural-multiscale-materials

//----------------------------------------------------------------------

// math
float compMax(float2 v) { return max(v.x, v.y); }
float maxNrm(float2 v) { return compMax(abs(v)); }
float2x2 inverse2(float2x2 m) {
    return float2x2(m[1][1], -m[0][1], -m[1][0], m[0][0]) / (m[0][0] * m[1][1] - m[0][1] * m[1][0]);
}

float erfinv(float x) {
    float w, p;
    w = -log((1.0-x)*(1.0+x));
    if(w < 5.000000) {
        w = w - 2.500000;
        p = 2.81022636e-08;
        p = 3.43273939e-07 + p*w;
        p = -3.5233877e-06 + p*w;
        p = -4.39150654e-06 + p*w;
        p = 0.00021858087 + p*w;
        p = -0.00125372503 + p*w;
        p = -0.00417768164 + p*w;
        p = 0.246640727 + p*w;
        p = 1.50140941 + p*w;
    }
    else {
        w = sqrt(w) - 3.000000;
        p = -0.000200214257;
        p = 0.000100950558 + p*w;
        p = 0.00134934322 + p*w;
        p = -0.00367342844 + p*w;
        p = 0.00573950773 + p*w;
        p = -0.0076224613 + p*w;
        p = 0.00943887047 + p*w;
        p = 1.00167406 + p*w;
        p = 2.83297682 + p*w;
    }
    return p*x;
}

// ray differentials
void calcDpDxy( in float3 ro, in vec3 rd, in vec3 rdx, in vec3 rdy, in float t, in vec3 nor, 
                out vec3 dpdx, out vec3 dpdy ) {
    dpdx = 2.*t*(rdx*dot(rd,nor)/dot(rdx,nor) - rd) * sign(dot(rd, rdx));
    dpdy = 2.*t*(rdy*dot(rd,nor)/dot(rdy,nor) - rd) * sign(dot(rd, rdy));
}

// some microfacet BSDF geometry factors
// (divided by NoL * NoV b/c cancelled out w/ microfacet BSDF)
float geometryFactor(float NoL, float NoV, float2 roughness) {
    float a2 = roughness.x * roughness.y;
    NoL = abs(NoL);
    NoV = abs(NoV);

    float G_V = NoV + sqrt((NoV - NoV * a2) * NoV + a2);
    float G_L = NoL + sqrt((NoL - NoL * a2) * NoL + a2);
    return 1. / (G_V * G_L);
}

int2 multilevelGridIdx(int2 idx) {
    return idx >> findLSB(idx);
}

//----------------------------------------------------------------------

// stable binomial 'random' numbers: interpolate between result for
// two closest binomial distributions where log_{.9}(p_i) integers
float binomial_interp(float u, float N, float p) {
    if(p >= 1.)
        return N;
    else if(p <= 1e-30)
        return 0.;

    // convert to distribution on ints while retaining expected value
    float cN = ceil(N);
    int iN = int(cN);
    p = p * (N / cN);
    N = cN;

    // round p to nearest powers of .9 (more stability)
    float pQ = .9;
    float pQef = log2(p) / log2(pQ);
    float p2 = exp2(floor(pQef) * log2(pQ));
    float p1 = p2 * pQ;
    float2 ps = float2(p1, p2);

    // compute the two corresponding binomials in parallel
    float2 pm = pow(1. - ps, float2(N));
    float2 cp = pm;
    float2 r = float2(N);

    float i = 0.0;
    // this should actually be < N, no dynamic loops in ShaderToy right now
    for(int ii = 0; ii <= 17; ++ii)
    {
        if(u < cp.x)
            r.x = min(i, r.x);
        if(u < cp.y) {
            r.y = i;
            break;
        }
        // fast path
        if(ii > 16)
        {
            float C = 1. / (1. - pow(p, N - i - 1.));
            float2 U = (u - cp) / (1. - cp);
            float2 A = (i + 1. + log2(1. - U / C) / log2(p));
            r = min(A, r);
            break;
        }

        i += 1.;
        pm /= 1. - ps;
        pm *= (N + 1. - i) / i;
        pm *= ps;
        cp += pm;
    }

    // interpolate between the two binomials according to log p (akin to mip interpolation)
    return mix(r.y, r.x, fract(pQef));
}
// resort to gaussian distribution for larger N*p
float approx_binomial(float u, float N, float p) {
    if (p * N > 5.)
    {
        float e = N * p;
        float v = N * p * max(1. - p, 0.0);
        float std = sqrt(v);
        float k = e + erfinv(mix(-.999999, .999999, u)) * std;
        return min(max(k, 0.), N);
    }
    else
        return binomial_interp(u, N, p);
}

//----------------------------------------------------------------------

vec3 glints(float2 texCO, float2 duvdx, float2 duvdy, mat3 ctf
  , vec3 lig, vec3 nor, vec3 view
  , float2 roughness, float2 microRoughness, float searchConeAngle, float variation, float dynamicRange, float density)
{
   vec3 col = vec3(0.);

    // Compute pixel footprint in texture space, step size w.r.t. anisotropy of the footprint
    float2x2 uvToPx = inverse2(float2x2(duvdx, duvdy));
    float2 uvPP = 1. / float2(maxNrm(uvToPx[0]), maxNrm(uvToPx[1]));

    // material
    float2 mesoRoughness = sqrt(max(roughness * roughness - microRoughness * microRoughness, float2(1.e-12))); // optimizer fail, max 0 removed

    // Anisotropic compression of the grid
    float2 texAnisotropy = float2( min(mesoRoughness.x / mesoRoughness.y, 1.)
                             , min(mesoRoughness.y / mesoRoughness.x, 1.) );

    // Compute half vector (w.r.t. dir light)
    vec3 hvW = normalize(lig + view);
    vec3 hv = normalize(hvW * ctf);
    float2 h = hv.xy / hv.z;
    float2 h2 = 0.75 * hv.xy / (hv.z + 1.);
    // Anisotropic compression of the slope-domain grid
    h2 *= texAnisotropy;

    // Compute the Gaussian probability of encountering a glint within a given finite cone
    float2 hppRScaled = h / roughness;
    float pmf = (microRoughness.x * microRoughness.y) / (roughness.x * roughness.y)
        * exp(-dot(hppRScaled, hppRScaled)); // planeplane h
    pmf /= hv.z * hv.z * hv.z * hv.z; // projected h
//  pmf /= dot(lig, nor) * dot(view, nor); // projected area, cancelled out by parts of G, ...
    float pmfToBRDF = 1. / (3.14159 * microRoughness.x * microRoughness.y);
    pmfToBRDF /= 4.; // solid angle o
    pmfToBRDF *= geometryFactor(dot(lig, nor), dot(view, nor), roughness); // ... see "geometryFactor"
    // phenomenological: larger cones flatten distribution
    float searchAreaProj = searchConeAngle * searchConeAngle / (4. * dot(lig, hvW) * hv.z); // * PI
    pmf = mix(pmf, 1., clamp(searchAreaProj, 0.0, 1.0)); // searchAreaProj / PI
    pmf = min(pmf, 1.);
    
    // noise coordinate (decorrelate interleaved grid)
    texCO += float2(100.);
    // apply anisotropy _after_ footprint estimation
    texCO *= texAnisotropy;

    // Compute AABB of pixel in texture space
    float2 uvAACB = max(abs(duvdx), abs(duvdy)) * texAnisotropy; // border center box
    float2 uvb = texCO - 0.5 * uvAACB;
    float2 uve = texCO + 0.5 * uvAACB;

    float2 uvLongAxis = uvAACB.x > uvAACB.y ? float2(1.0, 0.0) : float2(0.0, 1.0);
    float2 uvShortAxis = 1.0 - uvLongAxis;

    // Compute skew correction to snap axis-aligned line sampling back to longer anisotropic pixel axis in texture space
    float2 skewCorr2 = -(uvToPx * uvLongAxis) / (uvToPx * uvShortAxis);
    float skewCorr = abs((uvToPx * uvShortAxis).x) > abs((uvToPx * uvShortAxis).y) ? skewCorr2.x : skewCorr2.y;
    skewCorr *= dot(texAnisotropy, uvShortAxis) / dot(texAnisotropy, uvLongAxis);

    float isoUVPP = dot(uvPP, uvShortAxis);
    // limit anisotropy
    isoUVPP = max(isoUVPP, dot(uvAACB, uvLongAxis) / 16.0);

     // Two virtual grid mips: current and next
    float fracMip = log2(isoUVPP);
    float lowerMip = floor(fracMip);
    float uvPerLowerC = exp2(lowerMip);

    // Current mip level and cell size
    float uvPC = uvPerLowerC;
    float mip = lowerMip;

    int iter = 0;
    int iterThreshold = 150;

    for (int i = 0; i < 2; ++i)
    {
        float mipWeight = 1.0 - abs(mip - fracMip);

        float2 uvbg = min(uvb + 0.5 * uvPC, texCO);
        float2 uveg = max(uve - 0.5 * uvPC, texCO);

        // Snapped uvs of the cell centers
        float2 uvbi = floor(uvbg / uvPC);
        float2 uvbs = uvbi * uvPC;
        float2 uveo = uveg + uvPC - uvbs;

        // Resulting compositing values for a current layer
        float weight = 0.0;
        vec3 reflection = vec3(0.0);

        // March along the long axis
        float2 uvo = float2(0.0), uv = uvbs, uvio = float2(0.0), uvi = uvbi;
        for (int iter1 = 0; iter1 < 18; ++iter1) // horrible WebGL-compatible static for loop
        {
            // for cond:
            if (dot(uvo, uvLongAxis) < dot(uveo, uvLongAxis) && iter < iterThreshold);
            else break;

            // Snap samples to long anisotropic pixel axis
            float uvShortCenter = dot(texCO, uvShortAxis) + skewCorr * dot(uv - texCO, uvLongAxis);

            // Snapped uvs of the cell center
            uvi += (floor(uvShortCenter / uvPC) - dot(uvi, uvShortAxis)) * uvShortAxis;
            uv = uvi * uvPC;
            float uvShortEnd = uvShortCenter + uvPC;

            float2 uvb2 = uvbg * uvLongAxis + uvShortCenter * uvShortAxis;
            float2 uve2 = uveg * uvLongAxis + uvShortCenter * uvShortAxis;

            // March along the shorter axis
            for (int iter2 = 0; iter2 < 4; ++iter2) // horrible WebGL-compatible static for loop
            {
                // for cond:
                if (dot(uv, uvShortAxis) < uvShortEnd && iter < iterThreshold);
                else break;

                // Compute interleaved cell index
                int2 cellIdx = int2(uvi + float2(.5));
                cellIdx = multilevelGridIdx(cellIdx);

                // Randomize a glint based on a texture-space id of current grid cell
                float2 u2 = hash2(float( (cellIdx.x + 1549 * cellIdx.y) ));
                // Compute index of the cone
                float2 hg = h2 / (microRoughness + searchConeAngle);
                float2 hs = floor(hg + u2) + u2 * 533.;    // discrete cone index in paraboloid hv grid
                int2 coneIdx = int2(hs);

                // Randomize glint sizes within this layer
                float var_u = hash(float( (cellIdx.x + cellIdx.y * 763 + coneIdx.x + coneIdx.y * 577) ));
                float mls = 1. + variation * erfinv(mix(-.999, .999, var_u));
                if (mls <= 0.0) mls = fract(mls) / (1. - mls);
                mls = max(mls, 1.e-12);

                // Bilinear interpolation using coverage made by areas of two rects
                float2 mino = max(1.0 - max((uvb2 - uv) / uvPC, 0.0), 0.0);
                float2 maxo = max(1.0 - max((uv - uve2) / uvPC, 0.0), 0.0);
                float2 multo = mino * maxo;
                float coverageWeight = multo.x * multo.y;

                float cellArea = uvPC * uvPC;
                // Expected number of glints 
                float eN = density * cellArea;
                float sN = max(eN * mls, min(1.0, eN));
                eN = eN * mls;

                // Sample actually found number of glints
                float u = hash(float(coneIdx.x + coneIdx.y * 697));
                float lN = approx_binomial(u, sN, pmf);
#if 0
                // Colored glints
                if (false) {
                    vec3 glintColor = hue_colormap(fract(u + u2.y));
                    glintColor = mix(vec3(1.0f), glintColor, 1. / sqrt(max(lN, 1.0)));
                }
#endif
                // Ratio of glinting vs. expected number of microfacets
                float ratio = lN / eN;
                
                // limit dynamic range (snow more or less unlimited)
                ratio = min(ratio, dynamicRange * pmf);
                
                // convert to reflectance
                ratio *= pmfToBRDF;
#if 0
                // Grid
                reflection += vec3(u);
                weight += coverageWeight;
#else
                // Accumulate results
                reflection += coverageWeight * ratio;
                weight += coverageWeight;
#endif

                // for incr:
                uv += uvPC * uvShortAxis, uvi += uvShortAxis, ++iter;
            }

            // for incr:
              uvo += uvPC * uvLongAxis, uv = uvbs + uvo
            , uvio += uvLongAxis, uvi = uvbi + uvio;
        }

#ifdef DEBUG
        // Normalization
        if (weight < 1.e-15) {
            col = vec3(0.0, 1.0, 1.0);
            break;
        }
#endif

        reflection = reflection / weight;

        // Compositing of two layers
        col += mipWeight * reflection;

        // for incr:
        uvPC *= 2., mip += 1.;
    }

    return col;
}