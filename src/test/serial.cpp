// src/main.cpp
#include <Arduino.h>
#include <FreeRTOS.h>
#include <task.h>

void loggerTask(void* pvParameters) {
    for (uint32_t i = 0;; i++) {
        Serial.printf("LOG %lu | tick=%lu | core=%d | free_heap=%u\n",
                      i,
                      xTaskGetTickCount(),
                      portGET_CORE_ID(),
                      xPortGetFreeHeapSize());

        vTaskDelay(pdMS_TO_TICKS(2));
    }
}

void setup() {
    Serial.begin(2000000);

    while (!Serial) {
        ; // Do nothing — just busy-wait
    }

    Serial.println("=== Teensy 4.1 FreeRTOS serial alive ===");

    xTaskCreate(loggerTask, "Logger", 4096, nullptr, 1, nullptr);
    vTaskStartScheduler();
}

void loop() {
    // Required empty function for Teensyduino linker
}