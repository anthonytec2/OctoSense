#include <Wire.h>
#include "RTClib.h"

#define SQW_PIN 16 // RTC Clock Pin

RTC_DS3231 rtc;
const int PIN_IMU      = 2;   // IMU PPS output
const int PIN_RGB_L    = 5;   // 100Hz RGB left (continuous)
const int PIN_EVENT    = 8;   // Event camera PPS output (shares the wide pulse)
const int PIN_RGB_1HZ  = 24;  // RGB 1Hz PPS output
const int PIN_LIDAR    = 12;  // Lidar PPS output (shares the wide pulse)

// Constants
const int PPS_PULSE_TICKS = 2;        // 10ms pulse width for standard pulses

// Timers
IntervalTimer timer100Hz; // Generates 100 Hz synced to external PPS

// State variables
volatile uint16_t tickCount = 0;
volatile bool rgbState = LOW;
volatile bool ppsEventActive = false;          // Tracks first IMU/RGB_1HZ 10ms pulse window
volatile bool widePulseActive = false;         // Tracks EVENT/LIDAR wide pulse window
volatile bool secondPulseActive = false;
volatile uint8_t ppsIndex = 0;                 // Counts PPS pulses since START (1..), Note this is the rollover for UID resync
volatile uint16_t widePulseWidthTicks = 0;     // Per-PPS EVENT/LIDAR pulse width (ticks)
volatile bool secondPulseScheduled = false;    // Whether a second pulse is scheduled for IMU/RGB_1HZ this PPS
volatile uint16_t secondPulseStartTick = 0;    // Start tick for the second pulse (IMU/RGB_1HZ)
volatile uint16_t secondPulseEndTick = 0;      // End tick for the second pulse (IMU/RGB_1HZ)
volatile bool systemActive = false;
volatile bool rtcPulseDetected = false;
volatile uint8_t rtcTicksUntilStart = 0;       // Countdown: wait for N RTC ticks before activating PPS

void setup() {
  Serial.begin(115200);
  Serial.println("External PPS Generator Ready");
  Serial.println("Commands: START, STOP, STATUS");

  Wire.begin();
  rtc.begin();

  // Configure DS3231 SQW to 1Hz
  rtc.writeSqwPinMode(DS3231_SquareWave1Hz);

  // Configure pins
  pinMode(PIN_IMU, OUTPUT);
  pinMode(PIN_RGB_L, OUTPUT);
  pinMode(PIN_EVENT, OUTPUT);
  pinMode(PIN_RGB_1HZ, OUTPUT);
  pinMode(PIN_LIDAR, OUTPUT);
  pinMode(SQW_PIN, INPUT_PULLUP);

  // Initialize LOW
  digitalWriteFast(PIN_IMU, LOW);
  digitalWriteFast(PIN_RGB_L, LOW);
  digitalWriteFast(PIN_EVENT, LOW);
  digitalWriteFast(PIN_RGB_1HZ, LOW);
  digitalWriteFast(PIN_LIDAR, LOW);

  // Start 100Hz timer immediately for RGB_L (will be synced to PPS when START is issued)
  timer100Hz.begin(timer100HzISR, 5000); // 5ms intervals
  timer100Hz.priority(1);

  // Attach PPS interrupt immediately so we can always phase-align 100Hz to PPS
  attachInterrupt(digitalPinToInterrupt(SQW_PIN), rtcPulseISR, RISING);
}

void loop() {
  // Handle serial commands
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    command.toUpperCase();

    if (command == "START") {
      startPPSGenerator();
    }
    else if (command == "STOP") {
      stopPPSGenerator();
    }
    else if (command == "STATUS") {
      Serial.println(systemActive ? "Status: RUNNING" : "Status: STOPPED");
    }
    else {
      Serial.println("Unknown command. Use: START, STOP, or STATUS");
    }
  }

  // Check for RTC pulse (always align 100Hz to PPS)
  if (rtcPulseDetected) {
    rtcPulseDetected = false;

    if (systemActive) {
      // If still waiting for initial RTC ticks, count down
      if (rtcTicksUntilStart > 0) {
        rtcTicksUntilStart--;
        Serial.print("Waiting for RTC tick ");
        Serial.print(4 - rtcTicksUntilStart);
        Serial.println(" of 4...");

        // Once countdown reaches 0, start generating PPS pulses
        if (rtcTicksUntilStart == 0) {
          Serial.println("RTC synchronized - activating PPS pulses");
        }
      } else {
        // Normal PPS pulse generation
        handleExternalPPS();
      }
    } else {
      // Even when not active, keep 100Hz aligned to PPS
      ppsSyncOnly();
    }
  }
}

// Control functions
void startPPSGenerator() {
  if (!systemActive) {
    systemActive = true;
    rtcTicksUntilStart = 4;  // Wait for 4 RTC ticks before activating PPS

    // Attach interrupt to external RTC pulse (rising edge)
    attachInterrupt(digitalPinToInterrupt(SQW_PIN), rtcPulseISR, RISING);

    Serial.println("External PPS Generator STARTED - waiting for 4 RTC ticks");
  } else {
    Serial.println("Already running");
  }
}

void stopPPSGenerator() {
  if (systemActive) {
    systemActive = false;

    // Force PPS pins LOW (but leave RGB_L running)
    digitalWriteFast(PIN_IMU, LOW);
    digitalWriteFast(PIN_EVENT, LOW);
    digitalWriteFast(PIN_RGB_1HZ, LOW);
    digitalWriteFast(PIN_LIDAR, LOW);

    // Reset state
    tickCount = 0;
    rgbState = LOW;
    ppsEventActive = false;
    secondPulseActive = false;
    widePulseActive = false;
    ppsIndex = 0;
    widePulseWidthTicks = 0;
    secondPulseScheduled = false;
    secondPulseStartTick = 0;
    secondPulseEndTick = 0;
    rtcPulseDetected = false;
    rtcTicksUntilStart = 0;

    Serial.println("External PPS Generator STOPPED");
  } else {
    Serial.println("Already stopped");
  }
}

// RTC pulse interrupt handler (called on rising edge of external 1Hz signal)
void rtcPulseISR() {
  // Always capture PPS edges so we can keep timers aligned even when stopped
  rtcPulseDetected = true;
}

// Keep 100Hz aligned to PPS without generating PPS pulses on other pins
void ppsSyncOnly() {
  // Base level at PPS boundary
  rgbState = HIGH;
  digitalWriteFast(PIN_RGB_L, HIGH);

  // Re-sync 100Hz timer phase to PPS
  timer100Hz.end();
  timer100Hz.begin(timer100HzISR, 5000); // 5ms intervals
  timer100Hz.priority(1);
}

// Handle external PPS trigger
void handleExternalPPS() {
  // First, align continuous signal (100Hz) to this PPS edge
  ppsSyncOnly();

  // Immediately set all PPS pins HIGH (concurrent start)
  digitalWriteFast(PIN_IMU, HIGH);
  digitalWriteFast(PIN_EVENT, HIGH);
  digitalWriteFast(PIN_RGB_1HZ, HIGH);
  digitalWriteFast(PIN_LIDAR, HIGH);

  // Reset counters and state
  tickCount = 0;
  rgbState = HIGH;
  ppsEventActive = true;
  widePulseActive = true;
  secondPulseActive = false;

  // Increment PPS index (first PPS after START will be index 1)
  ppsIndex++;

  // Configure EVENT/LIDAR pulse width for first 6 PPS pulses: width = 20ms + 10ms * i (i = 1..6)
  // After that, revert to standard 10ms (2 ticks)
  if (ppsIndex <= 6) {
    widePulseWidthTicks = 4 + (2 * ppsIndex); // (4+2i) ticks => 30..80ms
  } else {
    widePulseWidthTicks = PPS_PULSE_TICKS; // standard 10ms
  }

  // For IMU and RGB_1HZ, schedule a second pulse for first 6 PPS pulses
  // Spacing is 20ms + 10ms * i measured from the END of the first pulse to the START of the second pulse
  if (ppsIndex <= 6) {
    secondPulseStartTick = PPS_PULSE_TICKS + 4 + (2 * ppsIndex); // end of first pulse + (4+2i) ticks spacing
    secondPulseEndTick = secondPulseStartTick + PPS_PULSE_TICKS; // 10ms duration
    secondPulseScheduled = true;
  } else {
    secondPulseScheduled = false;
  }

  // Timer has already been synchronized in ppsSyncOnly()
}

// 100Hz timer ISR
void timer100HzISR() {
  // Always toggle RGB_L for continuous 100Hz signal
  rgbState = !rgbState;
  digitalWriteFast(PIN_RGB_L, rgbState);

  // Only handle PPS-related timing if system is active
  if (!systemActive) return;

  tickCount++;

  // Handle EVENT/LIDAR wide pulse
  if (widePulseActive && tickCount >= widePulseWidthTicks) {
    digitalWriteFast(PIN_EVENT, LOW);
    digitalWriteFast(PIN_LIDAR, LOW);
    widePulseActive = false;
  }

  // Handle initial IMU and RGB_1HZ pulses after standard 10ms
  if (ppsEventActive && tickCount >= PPS_PULSE_TICKS) {
    digitalWriteFast(PIN_IMU, LOW);
    digitalWriteFast(PIN_RGB_1HZ, LOW);
    ppsEventActive = false;
  }

  // Start scheduled second pulse for IMU and RGB_1HZ, if any
  if (secondPulseScheduled && !secondPulseActive && tickCount == secondPulseStartTick) {
    digitalWriteFast(PIN_IMU, HIGH);
    digitalWriteFast(PIN_RGB_1HZ, HIGH);
    secondPulseActive = true;
  }

  // End the second pulse after its duration
  if (secondPulseActive && tickCount >= secondPulseEndTick) {
    digitalWriteFast(PIN_IMU, LOW);
    digitalWriteFast(PIN_RGB_1HZ, LOW);
    secondPulseActive = false;
  }
}
