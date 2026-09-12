"""
================================================================================
SkyGuard AI — Diagnostic Alert Card Generator
================================================================================
Generates human-readable, actionable diagnostic alert cards for IMD meteorologists
and AWS field maintenance engineers based on Explainable AI (XAI) outputs.
"""

from typing import Dict, Any, List, Optional, Union
import json

from .explainer import ExplanationResult
from anomaly_injection.taxonomy import AnomalyType


class DiagnosticCardGenerator:
    """
    Constructs standardized diagnostic alert cards and maintenance action work orders.
    """

    # Prescriptive maintenance recommendations mapped to specific failure modes & sensors
    ACTION_CATALOGUE = {
        AnomalyType.SPIKE.value: {
            "title": "Transient Electrical / Impulse Spike",
            "action": "Inspect sensor signal wiring, ADC ground reference, and lightning surge arrestor for electrical noise transients.",
            "urgency": "MEDIUM",
            "technician_task": "Check terminal block tightness and verify ADC ground potential < 0.5V.",
        },
        AnomalyType.DROP.value: {
            "title": "Intermittent Signal Drop / Power Sag",
            "action": "Inspect battery supply rail, solar charge controller, and signal cable for loose spade connections.",
            "urgency": "HIGH",
            "technician_task": "Measure solar battery terminal voltage under load; clean battery terminals.",
        },
        AnomalyType.FROZEN_SENSOR.value: {
            "title": "Sensor Signal Flatline / Bus Lockup",
            "action": "Power-cycle data logger and sensor interface module. Check RS-485 / SDI-12 communication bus for bus contention.",
            "urgency": "CRITICAL",
            "technician_task": "Reboot logger module, test serial bus termination resistor (120Ω), replace sensor transducer if unresponsive.",
        },
        AnomalyType.SENSOR_DRIFT.value: {
            "title": "Calibration Drift / Transducer Aging",
            "action": "Sensor calibration standard drift detected. Dispatch field calibration crew with secondary reference standard.",
            "urgency": "HIGH",
            "technician_task": "Perform multi-point field calibration against calibrated reference standard; replace PRT / capacitive film if offset > 1.5°C or 5% RH.",
        },
        AnomalyType.CONSTANT_OFFSET.value: {
            "title": "Static Zero-Point Bias / Physical Barrier",
            "action": "Inspect sensor radiation shield for biological nesting (wasp nests, dust) or physical obstruction; verify zero-offset potentiometer.",
            "urgency": "MEDIUM",
            "technician_task": "Dismantle and wash louvers of radiation shield; check barometer static port tubing for insect blockages.",
        },
        AnomalyType.HIGH_NOISE.value: {
            "title": "Degraded Signal-to-Noise Ratio",
            "action": "Check shielding braid continuity and ground loop currents. Inspect proximity of VHF/GPRS transmission antenna to sensor cables.",
            "urgency": "MEDIUM",
            "technician_task": "Ensure sensor cables run in separate conduit from telemetry RF cabling; verify chassis ground rod impedance < 5Ω.",
        },
        AnomalyType.MISSING_DATA.value: {
            "title": "Packet Dropout / Total Signal Loss",
            "action": "Verify telemetry modem connectivity, SIM card status, and data logger transmission schedule.",
            "urgency": "CRITICAL",
            "technician_task": "Inspect cable continuity between sensor head and logger channel; replace severed cable if resistance > 10Ω.",
        },
        AnomalyType.COMMUNICATION_CORRUPTION.value: {
            "title": "Serial Bus Frame Corruption / CRC Failure",
            "action": "Check RS-485 / Modbus parity and baud rate synchronization; replace EMI-degraded patch cable.",
            "urgency": "HIGH",
            "technician_task": "Inspect transceiver optocouplers and check for line reflection; replace shielded twisted pair cable.",
        },
        AnomalyType.PHYSICALLY_IMPOSSIBLE.value: {
            "title": "Gross Transducer Physical Envelope Rupture",
            "action": "Catastrophic transducer breakdown. Replace sensor probe immediately.",
            "urgency": "CRITICAL",
            "technician_task": "Immediate sensor head swap required. Return defective unit to central IMD instrumentation laboratory.",
        },
        AnomalyType.CROSS_SENSOR_INCONSISTENCY.value: {
            "title": "Psychrometric Thermodynamic Incoherence",
            "action": "Inspect relative humidity protective membrane (sintered cap) for water condensation or salt deposition.",
            "urgency": "HIGH",
            "technician_task": "Clean or replace RH sintered protective filter cap; recalibrate hygrometer chamber.",
        },
        AnomalyType.COORDINATED_MULTIVARIATE.value: {
            "title": "Multi-Sensor Coordinated Failure / Power Bus Failure",
            "action": "Station power rail instability affecting all channels simultaneously. Immediate power supply diagnostic required.",
            "urgency": "CRITICAL",
            "technician_task": "Replace 12V solar buffer battery and test 5V/3.3V internal voltage regulators on AWS mainboard.",
        },
    }

    @classmethod
    def generate_card(
        cls,
        explanation: ExplanationResult,
        severity: str = "ANOMALY",
        composite_score: float = 0.72,
    ) -> Dict[str, Any]:
        """
        Generate a complete, structured Diagnostic Alert Card.
        """
        cause = explanation.predicted_cause
        meta = cls.ACTION_CATALOGUE.get(cause, {
            "title": f"Unspecified Anomaly ({cause})",
            "action": "Perform general physical inspection of station sensors and logger.",
            "urgency": "MEDIUM",
            "technician_task": "Check logger diagnostic logs and verify sensor terminal connections.",
        })

        card_dict = {
            "alert_header": {
                "station_id": explanation.station_id,
                "timestamp": explanation.timestamp,
                "severity": severity,
                "composite_score": round(float(composite_score), 4),
                "predicted_fault": cause,
                "fault_title": meta["title"],
                "confidence": round(float(explanation.cause_confidence), 4),
                "urgency": meta["urgency"],
            },
            "sensor_attribution": {
                "primary_culprit": explanation.primary_culprit,
                "all_culprits": explanation.all_culprits,
                "details": {
                    k: {
                        "deviation_sigma": v.deviation_sigma,
                        "step_change": v.step_change,
                        "is_flatline": v.is_flatline,
                        "is_out_of_bounds": v.is_out_of_bounds,
                    }
                    for k, v in explanation.variable_attributions.items()
                },
            },
            "evidence_breakdown": {
                "summary_points": explanation.evidence_summary,
                "spatial_disparity": explanation.spatial_evidence,
                "thermodynamic_state": explanation.thermodynamic_evidence,
                "top_driving_features": explanation.top_contributing_features,
            },
            "maintenance_prescription": {
                "action": meta["action"],
                "technician_task": meta["technician_task"],
                "target_sensor": explanation.primary_culprit,
                "protocol": "IMD-AWS-STD-SOP-V4",
            },
        }

        return card_dict

    @classmethod
    def format_markdown(cls, card_dict: Dict[str, Any]) -> str:
        """
        Format a diagnostic card into clean, readable GitHub Markdown.
        """
        hdr = card_dict["alert_header"]
        attrib = card_dict["sensor_attribution"]
        evid = card_dict["evidence_breakdown"]
        maint = card_dict["maintenance_prescription"]

        lines = [
            f"# ⚠️ SkyGuard AI Diagnostic Alert — Station {hdr['station_id']}",
            f"**Timestamp:** `{hdr['timestamp']}` | **Severity:** `{hdr['severity']}` | **Score:** `{hdr['composite_score']:.3f}`",
            "",
            "## 🔍 Diagnostic Summary",
            f"- **Attributed Failure Mode:** `{hdr['predicted_fault']}` ({hdr['fault_title']})",
            f"- **Confidence:** `{hdr['confidence'] * 100:.1f}%`",
            f"- **Primary Sensor Culprit:** `{attrib['primary_culprit']}`",
            f"- **Maintenance Urgency:** `{hdr['urgency']}`",
            "",
            "## 📋 Evidence & Physical Discrepancies",
        ]

        for pt in evid["summary_points"]:
            lines.append(f"- {pt}")

        if evid["spatial_disparity"].get("has_disparity"):
            lines.append(f"- **Spatial Neighbor Disparity:** {evid['spatial_disparity']['description']}")

        if evid["thermodynamic_state"].get("has_violation"):
            lines.append(f"- **Psychrometric Violation:** {evid['thermodynamic_state']['description']}")

        lines.extend([
            "",
            "## 🛠️ Prescriptive Maintenance SOP",
            f"> **Standard Protocol:** `{maint['protocol']}`",
            f"> **Primary Action:** {maint['action']}",
            f"> **Field Technician Task:** {maint['technician_task']}",
            "",
        ])

        return "\n".join(lines)
