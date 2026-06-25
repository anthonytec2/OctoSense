class DataCollectionUI {
    constructor() {
        this.wsStatusEl = document.getElementById('ws-status');
        this.shutdownBtn = document.getElementById('shutdown-btn');
        this.driverStatusEl = document.getElementById('driver-status');

        this.bindEvents();
        this.startStatusPoller();
    }

    bindEvents() {
        document.getElementById('start-btn').addEventListener('click', () => this.startRecording());
        document.getElementById('stop-btn').addEventListener('click', () => this.stopRecording());
        if (this.shutdownBtn) {
            this.shutdownBtn.addEventListener('click', () => this.requestShutdown());
        }
    }

    startStatusPoller() {
        const poll = async () => {
            try {
                const response = await fetch('/api/status');
                if (response.ok) {
                    const data = await response.json();
                    this.updateConnectionStatus(true);
                    this.updateStatus(data);
                } else {
                    this.updateConnectionStatus(false);
                }
            } catch (err) {
                console.warn('Status poll failed', err);
                this.updateConnectionStatus(false);
            } finally {
                setTimeout(poll, 2000);
            }
        };
        poll();
    }

    updateConnectionStatus(connected) {
        this.wsStatusEl.textContent = connected ? 'Connected' : 'Disconnected';
        this.wsStatusEl.classList.toggle('status-good', connected);
        this.wsStatusEl.classList.toggle('status-bad', !connected);
    }

    updateStatus(status) {
        if (!status) return;
        document.getElementById('state').textContent = status.state ?? 'UNKNOWN';
        const disk = status.disk_space ?? {};
        const diskText = disk.available ? `${disk.available} free (${disk.percent} used)` : '--';
        document.getElementById('disk').textContent = diskText;

        // Toggle button visibility based on recording state
        const isRecording = status.state === 'RECORDING';
        document.getElementById('start-btn').style.display = isRecording ? 'none' : 'inline-block';
        document.getElementById('stop-btn').style.display = isRecording ? 'inline-block' : 'none';

        if (status.recording) {
            const elapsed = Math.floor(Date.now() / 1000 - status.recording.start_time);
            const minutes = Math.floor(elapsed / 60);
            const seconds = elapsed % 60;
            const timeStr = `${minutes}:${seconds.toString().padStart(2, '0')}`;
            document.getElementById('current-bag').textContent = `${status.recording.name} (${timeStr})`;
        } else {
            document.getElementById('current-bag').textContent = 'None';
        }

        if (status.sensors) {
            this.renderSensors(status.sensors);
        }
    }

    renderSensors(snapshot) {
        const tbody = document.querySelector('#sensor-table tbody');
        tbody.innerHTML = '';
        const sensors = snapshot.sensors || snapshot;

        // Sensor name mapping
        const nameMap = {
            'event_camera_0': 'EV-R',
            'event_camera_1': 'EV-L',
            'cam0': 'IMG-R',
            'cam1': 'IMG-L',
            'IR-73301414': 'IR',
            'vectornav': 'IMU',
            'can_bus': 'CAN',
            'ublox_gps': 'GPS',
            'rtcm': 'RTCM'
        };

        // Optional sensors that show gracefully when not present
        const optionalSensors = ['can_bus', 'ublox_gps', 'rtcm'];

        Object.entries(sensors).forEach(([name, meta]) => {
            const row = document.createElement('tr');
            const displayName = nameMap[name] || name;
            const deltaPct = meta.delta_pct != null ? `${(meta.delta_pct * 100).toFixed(2)}%` : 'N/A';
            const measured = meta.measured != null ? meta.measured.toFixed(2) : 'N/A';

            // Special handling for event cameras: green checkmark unless measured == 0, then red X
            let statusClass = `status-${meta.status || 'unknown'}`;
            let statusIcon = {
                ok: '✓',
                warning: '⚠',
                failed: '✗',
                unknown: '?',
            }[meta.status || 'unknown'];

            if (name === 'event_camera_0' || name === 'event_camera_1') {
                if (meta.measured !== null && meta.measured === 0) {
                    statusClass = 'status-failed';
                    statusIcon = '✗';
                } else {
                    statusClass = 'status-ok';
                    statusIcon = '✓';
                }
            }

            // Optional sensors: show OK if publishing, inactive if not
            if (optionalSensors.includes(name)) {
                if (meta.measured === 0 || meta.measured === null) {
                    statusClass = 'status-inactive';
                    statusIcon = '—';
                } else if (meta.measured > 0) {
                    // Any positive rate means the sensor is working
                    statusClass = 'status-ok';
                    statusIcon = '✓';
                }
            }

            row.innerHTML = `
        <td>${displayName}</td>
        <td>${measured}</td>
        <td class="${statusClass}">${statusIcon}</td>
        <td>${deltaPct}</td>
      `;
            tbody.appendChild(row);
        });
    }

    async startRecording() {
        const bagType = document.getElementById('bag-type').value;
        try {
            const response = await fetch('/api/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ bag_type: bagType }),
            });
            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.detail || result.message || 'Failed to start');
            }
            alert(`Recording started: ${result.bag.name}`);
        } catch (err) {
            alert(`Start failed: ${err.message}`);
        }
    }

    async stopRecording() {
        try {
            const response = await fetch('/api/stop', { method: 'POST' });
            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.detail || result.message || 'Failed to stop');
            }
            const duration = result.bag.duration?.toFixed(1) ?? '0';
            alert(`Recording stopped: ${result.bag.name} (${duration}s)`);
        } catch (err) {
            alert(`Stop failed: ${err.message}`);
        }
    }

    async requestShutdown() {
        if (!this.shutdownBtn) return;
        if (!window.confirm('Shut down the collection controller and exit the service?')) {
            return;
        }

        this.shutdownBtn.disabled = true;
        this.shutdownBtn.textContent = 'Shutting down…';
        this.setDriverStatusText('Shutting down…', 'status-running');

        try {
            const response = await fetch('/api/system/shutdown', { method: 'POST' });
            const result = await response.json().catch(() => ({}));
            if (!response.ok) {
                throw new Error(result.detail || result.message || 'Shutdown failed');
            }
        } catch (err) {
            alert(`Shutdown failed: ${err.message}`);
            this.shutdownBtn.disabled = false;
            this.shutdownBtn.textContent = 'Shutdown Controller';
            this.setDriverStatusText('Shutdown failed', 'status-failed');
        }
    }

    setDriverStatusText(text, statusClass) {
        if (!this.driverStatusEl) return;
        this.driverStatusEl.textContent = text;
        ['status-running', 'status-failed', 'status-succeeded'].forEach(cls => this.driverStatusEl.classList.remove(cls));
        if (statusClass) {
            this.driverStatusEl.classList.add(statusClass);
        }
    }

}

window.addEventListener('DOMContentLoaded', () => new DataCollectionUI());
