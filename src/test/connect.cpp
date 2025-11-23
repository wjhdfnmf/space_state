// src/main.cpp
#include <Arduino.h>
#include <FreeRTOS.h>
#include <task.h>

void echoTask(void* pvParameters) {
    char buffer[128];
    uint8_t index = 0;

    for (;;) {
        // Non-blocking read — runs at max USB speed
        while (Serial.available()) {
            char c = Serial.read();
            if (c == '\n' || c == '\r' || index >= sizeof(buffer)-1) {
                buffer[index] = '\0';
                // Convert to uppercase
                for (uint8_t i = 0; i < index; i++) {
                    buffer[i] = toupper(buffer[i]);
                }
                Serial.printf("%s\n", buffer);
                index = 0;
            } else {
                buffer[index++] = c;
            }
        }
        vTaskDelay(1);  // yield to other tasks
    }
}

void setup() {
    Serial.begin(2000000);
    while (!Serial) ;  // wait for host

    Serial.println("Teensy ready — send text, get UPPERCASE reply");

    xTaskCreate(echoTask, "Echo", 4096, nullptr, 2, nullptr);
    vTaskStartScheduler();
}

void loop() {
    // Required empty function
}