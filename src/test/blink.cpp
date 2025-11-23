#include <Arduino.h>
#include <FreeRTOS.h>
#include <task.h>

#define LED_PIN 13

void blinkTask(void* pvParameters) {
    pinMode(LED_PIN, OUTPUT);

    for (;;) {
        digitalWrite(LED_PIN, HIGH);   // Use normal digitalWrite (safe)
        vTaskDelay(pdMS_TO_TICKS(500));
        digitalWrite(LED_PIN, LOW);
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

void setup() {
    Serial.begin(2000000);

    // CORRECT: wait for USB serial without yielding
    while (!Serial) {
        ; // Do nothing — just busy-wait
        // Do NOT use vTaskDelay() here — scheduler not started yet!
    }

    Serial.println("=== Teensy 4.1 FreeRTOS alive ===");

    xTaskCreate(blinkTask, "Blink", 512, nullptr, 1, nullptr);
    vTaskStartScheduler();

    // Never reaches here
    for (;;) {}
}

void loop() {
    // Required empty function for Teensyduino linker
}