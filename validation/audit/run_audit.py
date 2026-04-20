from __future__ import annotations

from .common import AUDIT_REPORT_MD, AuditConfig, download_reference_dataset
from .diff_report import build_diff_report
from .pine_replica import run_pine_replica
from .python_runner import run_python_audit


def main() -> None:
    config = AuditConfig()
    dataset_df = download_reference_dataset(config)
    py = run_python_audit()
    pine = run_pine_replica()
    result = build_diff_report(
        dataset_df=dataset_df,
        python_signals=py.signals,
        pine_signals=pine.signals,
        python_per_bar=py.per_bar,
        pine_per_bar=pine.per_bar,
    )
    print(f"AUDIT_REPORT written to {AUDIT_REPORT_MD}")
    print(f"Verdict: {result['verdict']} | Match rate: {result['match_rate']:.2f}%")


if __name__ == "__main__":
    main()
