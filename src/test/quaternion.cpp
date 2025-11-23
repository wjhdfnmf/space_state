#include <Arduino.h>
#include <arm_math.h>

struct Quaternion {
    float w, x, y, z;

    // Default constructor = identity
    Quaternion() : w(1.0f), x(0.0f), y(0.0f), z(0.0f) {}

    // Constructor from components
    Quaternion(float qw, float qx, float qy, float qz) : w(qw), x(qx), y(qy), z(qz) {}

    // Normalize (critical after integration)
    void normalize() {
        float norm = arm_sqrt_f32(w*w + x*x + y*y + z*z);
        if (norm > 0.0f) {
            w /= norm; x /= norm; y /= norm; z /= norm;
        }
    }

    // Multiply: q1 * q2 (Hamilton product)
    Quaternion operator*(const Quaternion& q) const {
        return Quaternion(
            w*q.w - x*q.x - y*q.y - z*q.z,
            w*q.x + x*q.w + y*q.z - z*q.y,
            w*q.y - x*q.z + y*q.w + z*q.x,
            w*q.z + x*q.y - y*q.x + z*q.w
        );
    }

    // Rotate a vector (v' = q * v * q⁻¹)
    void rotateVector(float v[3], float result[3]) const {
        // q * v
        float vq[4] = {0, v[0], v[1], v[2]};
        Quaternion temp = *this * Quaternion(0, v[0], v[1], v[2]);
        Quaternion conj(w, -x, -y, -z);
        Quaternion rotated = temp * conj;

        result[0] = rotated.x;
        result[1] = rotated.y;
        result[2] = rotated.z;
    }

    // Integrate gyro (small-angle approximation, 1st order)
    void integrateGyro(const float gyro[3], float dt) {
        float wx = gyro[0], wy = gyro[1], wz = gyro[2];
        float half_dt = 0.5f * dt;

        Quaternion deltaQ(
            1.0f,
            half_dt * wx,
            half_dt * wy,
            half_dt * wz
        );

        *this = *this * deltaQ;
        normalize();  // prevent drift
    }

    // Convert to Euler angles (rad) — yaw/pitch/roll (321 sequence)
    void toEuler(float& roll, float& pitch, float& yaw) const {
        float test = x*y + z*w;
        if (test > 0.499f) { // singularity at north pole
            pitch = 2 * atan2(x, w);
            yaw   = PI/2;
            roll  = 0;
            return;
        }
        if (test < -0.499f) { // south pole
            pitch = -2 * atan2(x, w);
            yaw   = -PI/2;
            roll  = 0;
            return;
        }
        float sqx = x*x, sqy = y*y, sqz = z*z;
        roll  = atan2f(2*(w*x + y*z), 1 - 2*(sqx + sqy));
        pitch = asinf(2*test);
        yaw   = atan2f(2*(w*y + x*z), 1 - 2*(sqy + sqz));
    }
};

// Global attitude
Quaternion attitude;

// In your main loop or task
void loop() {
    static uint32_t lastMicros = 0;
    uint32_t now = micros();
    float dt = (now - lastMicros) * 1e-6f;
    lastMicros = now;

    float gyro[3] = {gyroX_rad_s, gyroY_rad_s, gyroZ_rad_s};  // from your IMU
    attitude.integrateGyro(gyro, dt);

    float rpy[3];
    attitude.toEuler(rpy[0], rpy[1], rpy[2]);
    Serial.printf("RPY: %.1f %.1f %.1f\n", 
                  rpy[0]*RAD_TO_DEG, rpy[1]*RAD_TO_DEG, rpy[2]*RAD_TO_DEG);
}