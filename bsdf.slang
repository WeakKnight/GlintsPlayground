static const float M_PI = 3.14159265;
static const float M_2PI = 6.28318531;  // 2pi
static const float M_4PI = 12.56637061;  // 4pi
static const float M_PI_2 = 1.57079632679489661923; // pi/2
static const float M_PI_4 = 0.785398163397448309616; // pi/4
static const float M_1_PI = 0.318309886183790671538; // 1/pi
static const float kMinCosTheta = 1e-6;

/*
*  Generate a vector that is orthogonal to the input vector.
*  This can be used to invent a tangent frame for meshes that don't have real tangents/bitangents.
*  \param u Unit vector.
*  \return v Unit vector that is orthogonal to u.
*/
float3 perp_stark(float3 u)
{
    // TODO: Validate this and look at numerical precision etc. Are there better ways to do it?
    float3 a   = abs(u);
    uint   uyx = (a.x - a.y) < 0 ? 1 : 0;
    uint   uzx = (a.x - a.z) < 0 ? 1 : 0;
    uint   uzy = (a.y - a.z) < 0 ? 1 : 0;
    uint   xm  = uyx & uzx;
    uint   ym  = (1 ^ xm) & uzy;
    uint   zm  = 1 ^ (xm | ym); // 1 ^ (xm & ym)
    float3 v   = normalize(cross(u, float3(xm, ym, zm)));
    return v;
}

struct Frame
{
    __init(float3 normalW)
    {
        N = normalW;
        T = perp_stark(N);
        B = cross(N, T);
    }

    __init(float3 normalW, float4 tangentW)
    {
        // Check that tangent space exists and can be safely orthonormalized.
        // Otherwise invent a tanget frame based on the normal.
        // We check that:
        //  - Tangent exists, this is indicated by a nonzero sign (w).
        //  - It has nonzero length. Zeros can occur due to interpolation or bad assets.
        //  - It is not parallel to the normal. This can occur due to normal mapping or bad assets.
        //  - It does not have NaNs. These will propagate and trigger the fallback.
        N = normalW;

        float NdotT       = dot(tangentW.xyz, N);
        bool  nonParallel = abs(NdotT) < 0.9999f;
        bool  nonZero     = dot(tangentW.xyz, tangentW.xyz) > 0.f;

        bool valid = tangentW.w != 0.f && nonZero && nonParallel;
        if (valid)
        {
            T = normalize(tangentW.xyz - N * NdotT);
            B = cross(N, T) * tangentW.w;
        }
        else
        {
            T = perp_stark(N);
            B = cross(N, T);
        }
    }

    float3 N;                 ///< Shading normal at shading hit.
    float3 T;                 ///< Shading tangent at shading hit.
    float3 B;                 ///< Shading bitangent at shading hit.

    /*
    *  Transform vector from the local surface frame to world space.
    *  \param v: Vector in local space.
    *  \return: Vector in world space.
    */
    float3 fromLocal(float3 v)
    {
        return T * v.x + B * v.y + N * v.z;
    }

    /*
    * Transform vector from world space to the local surface frame.
    * \param v: Vector in world space.
    * \return: Vector in local space.
    */
    float3 toLocal(float3 v)
    {
        return float3(dot(v, T), dot(v, B), dot(v, N));
    }
};

/** Evaluates the GGX (Trowbridge-Reitz) normal distribution function (D).

    Introduced by Trowbridge and Reitz, "Average irregularity representation of a rough surface for ray reflection", Journal of the Optical Society of America, vol. 65(5), 1975.
    See the correct normalization factor in Walter et al. https://dl.acm.org/citation.cfm?id=2383874
    We use the simpler, but equivalent expression in Eqn 19 from http://blog.selfshadow.com/publications/s2012-shading-course/hoffman/s2012_pbs_physics_math_notes.pdf

    For microfacet models, D is evaluated for the direction h to find the density of potentially active microfacets (those for which microfacet normal m = h).
    The 'alpha' parameter is the standard GGX width, e.g., it is the square of the linear roughness parameter in Disney's BRDF.
    Note there is a singularity (0/0 = NaN) at NdotH = 1 and alpha = 0, so alpha should be clamped to some epsilon.

    \param[in] alpha GGX width parameter (should be clamped to small epsilon beforehand).
    \param[in] cosTheta Dot product between shading normal and half vector, in positive hemisphere.
    \return D(h)
*/
float evalNdfGGX(float alpha, float cosTheta)
{
    float a2 = alpha * alpha;
    float d = ((cosTheta * a2 - cosTheta) * cosTheta + 1);
    return a2 / (d * d * M_PI);
}

/** Evaluates the Smith lambda function for the GGX normal distribution.
    See Eq 72 in http://jcgt.org/published/0003/02/03/paper.pdf

    \param[in] alphaSqr Squared GGX width parameter.
    \param[in] cosTheta Dot product between shading normal and the evaluated direction, in the positive hemisphere.
*/
float evalLambdaGGX(float alphaSqr, float cosTheta)
{
    if (cosTheta <= 0) return 0;
    float cosThetaSqr = cosTheta * cosTheta;
    float tanThetaSqr = max(1 - cosThetaSqr, 0) / cosThetaSqr;
    return 0.5 * (-1 + sqrt(1 + alphaSqr * tanThetaSqr));
}

/** Evaluates the height-correlated form of the masking-shadowing function for the GGX normal distribution, using Smith's approximation.
    See Eq 99 in http://jcgt.org/published/0003/02/03/paper.pdf

    Eric Heitz recommends using it in favor of the separable form as it is more accurate and of similar complexity.
    The function is only valid for cosThetaI > 0 and cosThetaO > 0  and should be clamped to 0 otherwise.

    \param[in] alpha GGX width parameter (should be clamped to small epsilon beforehand).
    \param[in] cosThetaI Dot product between shading normal and incident direction, in positive hemisphere.
    \param[in] cosThetaO Dot product between shading normal and outgoing direction, in positive hemisphere.
    \return G(cosThetaI, cosThetaO)
*/
float evalMaskingSmithGGXCorrelated(float alpha, float cosThetaI, float cosThetaO)
{
    float alphaSqr = alpha * alpha;
    float lambdaI = evalLambdaGGX(alphaSqr, cosThetaI);
    float lambdaO = evalLambdaGGX(alphaSqr, cosThetaO);
    return 1 / (1 + lambdaI + lambdaO);
}

/** Evaluates the Fresnel term using Schlick's approximation.
    Introduced in http://www.cs.virginia.edu/~jdl/bib/appearance/analytic%20models/schlick94b.pdf

    The Fresnel term equals f0 at normal incidence, and approaches f90=1.0 at 90 degrees.
    The formulation below is generalized to allow both f0 and f90 to be specified.

    \param[in] f0 Specular reflectance at normal incidence (0 degrees).
    \param[in] f90 Reflectance at orthogonal incidence (90 degrees), which should be 1.0 for specular surface reflection.
    \param[in] cosTheta Cosine of angle between microfacet normal and incident direction (LdotH).
    \return Fresnel term.
*/
float3 evalFresnelSchlick(float3 f0, float3 f90, float cosTheta)
{
    return f0 + (f90 - f0) * pow(max(1 - cosTheta, 0), 5); // Clamp to avoid NaN if cosTheta = 1+epsilon
}

float evalFresnelSchlick(float f0, float f90, float cosTheta)
{
    return f0 + (f90 - f0) * pow(max(1 - cosTheta, 0), 5); // Clamp to avoid NaN if cosTheta = 1+epsilon
}