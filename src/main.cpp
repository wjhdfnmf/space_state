#include <Arduino.h>
#undef B0
#undef B1
#undef B2
#undef B3
#undef B4
#undef B5
#undef B6
#undef B7

#include <FreeRTOS.h>
#include <task.h>
#include <queue.h>
#include <SD.h>
#include <SPI.h>

#include <arm_math.h>

// Static pool of packet buffers (adjust size as needed)
#define PACKET_POOL_SIZE 20
struct Packet { double data[7]; };
static Packet packetPool[PACKET_POOL_SIZE];
static uint8_t poolIndex = 0;

// Logging buffer and counters
static char logBuffer[128 * 100];  // ~12 KB, 100 lines max 128 chars each
static uint16_t logCount = 0;
static TickType_t lastFlushTime = 0;

QueueHandle_t dataQueue;
File logFile;

void receiveTask(void* pvParameters) {
    for (;;) {
        if (Serial.available() >= 56) {
            Packet* pkt = &packetPool[poolIndex++];
            if (poolIndex >= PACKET_POOL_SIZE) poolIndex = 0;

            Serial.readBytes((uint8_t*)pkt->data, 56);
            xQueueSend(dataQueue, &pkt, portMAX_DELAY);
        }
        vTaskDelay(1);  // prevent watchdog starvation
    }
}

void guidanceTask(void* pvParameters) {
    Packet* pkt;
    for (;;) {
        if (xQueueReceive(dataQueue, &pkt, portMAX_DELAY) == pdPASS) {
            // Use pkt->data[0] … pkt->data[6]
            Serial.printf("t=%.3f pos=(%.0f,%.0f,%.0f) vel=(%.0f,%.0f,%.0f)\n",
                          pkt->data[0], pkt->data[1], pkt->data[2], pkt->data[3],
                          pkt->data[4], pkt->data[5], pkt->data[6]);
            // Return buffer to pool (simple ring for this example)
            // In real code use a free-queue or semaphore

            // Fast append to RAM buffer
            logCount += snprintf(logBuffer + logCount, sizeof(logBuffer) - logCount,
                                 "%.6f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\r\n",
                                 pkt->data[0], pkt->data[1], pkt->data[2], pkt->data[3],
                                 pkt->data[4], pkt->data[5], pkt->data[6]);

            // Time-based flush (100 ms) OR immediate if buffer nearly full
            TickType_t now = xTaskGetTickCount();
            if ((logCount > 10000) || (now - lastFlushTime > pdMS_TO_TICKS(100))) {
                if (logCount > 0) {
                    logFile.write((uint8_t*)logBuffer, logCount);
                    logCount = 0;
                }
                logFile.flush();
                lastFlushTime = now;
            }
        }
    }
}

void setup() {
    Serial.begin(2000000);
    while (!Serial) {}

    if (!SD.begin(BUILTIN_SDCARD)) {
        Serial.println("SD card failed");
        while (1) vTaskDelay(1000);
    }
    logFile = SD.open("guidance.csv", FILE_WRITE);
    if (!logFile) {
        Serial.println("Failed to open log file");
        while (1) vTaskDelay(1000);
    }
    logFile.println("time,x,y,z,vx,vy,vz");  // header

    dataQueue = xQueueCreate(PACKET_POOL_SIZE, sizeof(Packet*));  // queue of pointers

    if (xTaskCreate(receiveTask, "Receive", 4096, nullptr, 3, nullptr) == pdPASS) {
        Serial.println("Receive task created");
    } else {
        Serial.println("Failed to create receive task");
    }
    if (xTaskCreate(guidanceTask, "Guidance", 8192, nullptr, 2, nullptr) == pdPASS) {
        Serial.println("Guidance task created");
    } else {
        Serial.println("Failed to create guidance task");
    }
    
    vTaskStartScheduler();
}

void loop() {
    // Intentionally empty — FreeRTOS scheduler runs forever
    // Do not put any code here
}