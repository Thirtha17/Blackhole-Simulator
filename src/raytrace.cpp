#include "raytrace.hpp"
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace py = pybind11;

// -------------------- helpers --------------------
static inline double clamp01(double x) { return std::max(0.0, std::min(1.0, x)); }
static inline double fract(double x) { return x - std::floor(x); }

struct Vec3 {
    double x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(double X, double Y, double Z) : x(X), y(Y), z(Z) {}
};

static inline Vec3 operator+(const Vec3& a, const Vec3& b) { return Vec3(a.x+b.x, a.y+b.y, a.z+b.z); }
static inline Vec3 operator-(const Vec3& a, const Vec3& b) { return Vec3(a.x-b.x, a.y-b.y, a.z-b.z); }
static inline Vec3 operator*(const Vec3& a, double s) { return Vec3(a.x*s, a.y*s, a.z*s); }
static inline Vec3 operator*(double s, const Vec3& a) { return Vec3(a.x*s, a.y*s, a.z*s); }
static inline Vec3 operator/(const Vec3& a, double s) { return Vec3(a.x/s, a.y/s, a.z/s); }

static inline double dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );
}
static inline double norm(const Vec3& a) { return std::sqrt(dot(a,a)); }

static inline Vec3 normalize(const Vec3& a) {
    double n = norm(a);
    if (n < 1e-14) return Vec3(0,0,0);
    return a / n;
}

// -------------------- procedural disk texture --------------------
static inline double hash2(double x, double y) {
    double h = std::sin(x*127.1 + y*311.7) * 43758.5453123;
    return fract(h);
}

// 0..1 thin bright streaks along azimuth
static inline double streaks(double r, double phi) {
    double a = std::sin(45.0*phi + 7.0*r);
    double b = std::sin(105.0*phi + 13.0*r);
    double n = 0.55*a + 0.45*b;

    double noise = hash2(r*2.7, phi*6.3);
    n = 0.82*n + 0.18*(2.0*noise - 1.0);

    n = std::fabs(n);
    n = std::pow(n, 10.0); // ↑ increase for thinner filaments
    return clamp01(n);
}

// -------------------- optical Schwarzschild (isotropic coords) --------------------
//
// Treat Schwarzschild as a gradient-index medium in isotropic coords.
// Ray equations (gradient-index optics):
//   dr/ds = v
//   dv/ds = ∇ln n - (v·∇ln n) v
//
// with refractive index:
//   n(rho) = (1 + M/(2 rho))^3 / (1 - M/(2 rho))
//

static inline Vec3 grad_ln_n(const Vec3& r, double M) {
    double rho = norm(r);
    if (rho < 1e-10) return Vec3(0,0,0);

    double q = M / (2.0 * rho);
    double one_p = 1.0 + q;
    double one_m = std::max(1.0 - q, 1e-6);

    // ln n = 3 ln(1+q) - ln(1-q),  q = M/(2 rho), dq/drho = -q/rho
    double dq = -q / rho;
    double dlnn_drho = dq * ( 3.0/one_p + 1.0/one_m );

    Vec3 rhat = r / rho;
    return rhat * dlnn_drho;
}

struct State {
    Vec3 r;
    Vec3 v; // unit direction
};

static inline State deriv(const State& s, double M) {
    State ds;
    ds.r = s.v;

    Vec3 g = grad_ln_n(s.r, M);
    double proj = dot(s.v, g);
    ds.v = g - s.v * proj;
    return ds;
}

static inline State rk4_step(const State& s, double h, double M) {
    State k1 = deriv(s, M);

    State s2; s2.r = s.r + k1.r*(h*0.5); s2.v = normalize(s.v + k1.v*(h*0.5));
    State k2 = deriv(s2, M);

    State s3; s3.r = s.r + k2.r*(h*0.5); s3.v = normalize(s.v + k2.v*(h*0.5));
    State k3 = deriv(s3, M);

    State s4; s4.r = s.r + k3.r*h;       s4.v = normalize(s.v + k3.v*h);
    State k4 = deriv(s4, M);

    State out;
    out.r = s.r + (k1.r + 2.0*k2.r + 2.0*k3.r + k4.r) * (h/6.0);
    out.v = normalize(s.v + (k1.v + 2.0*k2.v + 2.0*k3.v + k4.v) * (h/6.0));
    return out;
}

// -------------------- renderer --------------------
py::array_t<std::uint8_t> render_sky_with_bh(
    int width, int height, double fov_deg,
    double bh_angular_radius_deg,
    double lens_strength,
    py::array_t<std::uint8_t> sky_img) // unused; black background
{
    (void)sky_img;
    auto out = py::array_t<std::uint8_t>({height, width, 4});
    auto img = out.mutable_unchecked<3>();

    const double fov = fov_deg * M_PI / 180.0;
    const double aspect = (double)width / (double)height;
    const double warp = std::clamp(lens_strength, 0.0, 1.0);

    // Mass scale in geometric units (G=c=1).
    const double M = 0.8 + 1.4 * warp;
    const double rho_horizon = 0.5 * M;      // isotropic horizon
    const double rho_capture = 1.03 * rho_horizon;
    const double rho_escape = 70.0 * M;

    // Thin circular accretion disk in y=0 plane.
    const double disk_inner = 5.5 * M;
    const double disk_outer = 20.0 * M;
    const double disk_half_thickness = 0.08 * M;

    // Camera setup (slightly above disk plane, looking at origin).
    Vec3 cam(0.0, 2.8 * M, 27.0 * M);
    Vec3 look_at(0.0, 0.0, 0.0);
    Vec3 world_up(0.0, 1.0, 0.0);
    Vec3 forward = normalize(look_at - cam);
    Vec3 right = normalize(cross(forward, world_up));
    Vec3 up = normalize(cross(right, forward));

    const int max_steps = 1100;
    const double h_base = 0.030 * M;

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            double sx = (2.0 * i / (double)width - 1.0);
            double sy = (1.0 - 2.0 * j / (double)height);

            double tan_half = std::tan(0.5 * fov);
            Vec3 dir = normalize(
                forward +
                right * (sx * aspect * tan_half) +
                up * (sy * tan_half)
            );

            State s;
            s.r = cam;
            s.v = dir;

            double rr = 0.0, gg = 0.0, bb = 0.0;
            int disk_hits = 0;
            bool was_in_slab = false;
            double rho_min = 1e30;
            Vec3 prev_r = s.r;

            for (int step = 0; step < max_steps; step++) {
                double rho = norm(s.r);
                rho_min = std::min(rho_min, rho);

                if (rho < rho_capture) {
                    // If nothing was emitted before capture, the pixel is black.
                    // Otherwise keep foreground-disk light already accumulated.
                    if (disk_hits == 0) rr = gg = bb = 0.0;
                    break;
                }
                if (rho > rho_escape) {
                    break; // escaped to background
                }

                double h = h_base;
                if (rho > 10.0 * M) h *= 1.8;
                if (rho > 18.0 * M) h *= 2.8;
                if (rho > 35.0 * M) h *= 4.2;

                prev_r = s.r;
                State next = rk4_step(s, h, M);

                // Detect crossing through thin disk slab around y=0.
                double y1 = prev_r.y;
                double y2 = next.r.y;
                bool slab_possible =
                    (std::fabs(y1) < disk_half_thickness) ||
                    (std::fabs(y2) < disk_half_thickness) ||
                    (y1 * y2 < 0.0);

                if (slab_possible) {
                    double t = 0.0;
                    double denom = (y2 - y1);
                    if (std::fabs(denom) > 1e-12) t = (0.0 - y1) / denom;
                    t = std::clamp(t, 0.0, 1.0);

                    Vec3 p = prev_r + (next.r - prev_r) * t;
                    bool in_slab_now = (std::fabs(p.y) < disk_half_thickness);

                    if (in_slab_now && !was_in_slab) {
                        double r_disk = std::sqrt(p.x*p.x + p.z*p.z);
                        if (r_disk > disk_inner && r_disk < disk_outer) {
                            // Circular orbital flow around y-axis.
                            Vec3 vphi = normalize(Vec3(-p.z, 0.0, p.x));
                            Vec3 to_cam = normalize(cam - p);

                            // Simple relativistic orbital speed profile (capped).
                            double beta = std::sqrt(std::max(M / std::max(r_disk, 1e-6), 0.0));
                            beta = std::min(beta, 0.58);
                            double gamma = 1.0 / std::sqrt(std::max(1.0 - beta*beta, 1e-8));
                            double mu = dot(vphi, to_cam);
                            double doppler = 1.0 / (gamma * std::max(1.0 - beta * mu, 1e-4));

                            // Approx gravitational redshift in Schwarzschild-like form.
                            double ggrav = std::sqrt(std::max(1.0 - (2.0 * M) / std::max(r_disk, 2.05*M), 1e-4));

                            // Disk emissivity profile.
                            double emiss = std::pow(disk_inner / r_disk, 2.15);
                            double I = 8.5 * emiss * std::pow(doppler * ggrav, 3.0);

                            // Warm color ramp: inner hotter.
                            double hot = clamp01((disk_inner * 1.8 - r_disk) / (disk_inner * 1.1));
                            double t = clamp01((r_disk - disk_inner) / (disk_outer - disk_inner));
                            // Sky blue (inner) -> dark blue (outer)
                            double R = 170.0 - 150.0 * t;
                            double G = 220.0 - 160.0 * t;
                            double B = 255.0 - 110.0 * t;

                            double mult = (disk_hits == 0) ? 1.0 : 1.45;
                            rr += mult * R * I;
                            gg += mult * G * I;
                            bb += mult * B * I;
                            disk_hits++;
                            if (disk_hits >= 3) break;
                        }
                    }
                    was_in_slab = in_slab_now;
                } else {
                    was_in_slab = false;
                }

                s = next;
            }

            // Subtle photon ring boost from closest approach.
            {
                double ring_target = 2.9 * M;
                double ring_w = 0.45 * M;
                double d = (rho_min - ring_target) / std::max(ring_w, 1e-6);
                double glow = std::exp(-d*d);
                rr += 8.0 * glow;
                gg += 14.0 * glow;
                bb += 26.0 * glow;
            }

            rr = std::min(255.0, rr);
            gg = std::min(255.0, gg);
            bb = std::min(255.0, bb);
            img(j,i,0) = (std::uint8_t)rr;
            img(j,i,1) = (std::uint8_t)gg;
            img(j,i,2) = (std::uint8_t)bb;
            img(j,i,3) = 255;
        }
    }

    return out;
}
