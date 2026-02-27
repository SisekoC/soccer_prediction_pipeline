# presentation/presenter.py
"""Formats pipeline results for various output destinations and formats."""

import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


class Presenter:
    """
    Formats pipeline results for different output formats and destinations.
    Supports JSON, human-readable text, CSV, and Markdown.
    """

    def __init__(self, output_format: str = "json", output_file: Optional[str] = None):
        """
        Args:
            output_format: One of 'json', 'human', 'csv', 'markdown'.
            output_file: Optional file path to write output. If None, output is returned as string.
        """
        self.output_format = output_format.lower()
        self.output_file = Path(output_file) if output_file else None

    def format(
        self,
        home_team: str,
        away_team: str,
        match_date: Optional[str],
        features: pd.Series,
        prediction: Optional[np.ndarray] = None,
        actual: Optional[str] = None,
        validation: Optional[Dict] = None,
    ) -> Union[str, None]:
        """
        Format the complete pipeline output.

        Args:
            home_team: Home team name.
            away_team: Away team name.
            match_date: Match date (YYYY-MM-DD) or None.
            features: Pandas Series of 40 anchor scores.
            prediction: Optional probability array from model.
            actual: Optional actual result for validation.
            validation: Optional validation result dict.

        Returns:
            Formatted string if output_file is None, otherwise writes to file and returns None.
        """
        # Build the result dictionary
        result = self._build_result_dict(
            home_team, away_team, match_date, features, prediction, actual, validation
        )

        # Format according to selected output format
        if self.output_format == "json":
            output = self._format_json(result)
        elif self.output_format == "human":
            output = self._format_human(result)
        elif self.output_format == "csv":
            output = self._format_csv(result)
        elif self.output_format == "markdown":
            output = self._format_markdown(result)
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")

        # Write to file or return string
        if self.output_file:
            self._write_to_file(output)
            return None
        else:
            return output

    def _build_result_dict(
        self,
        home_team: str,
        away_team: str,
        match_date: Optional[str],
        features: pd.Series,
        prediction: Optional[np.ndarray],
        actual: Optional[str],
        validation: Optional[Dict],
    ) -> Dict[str, Any]:
        """Build a structured dictionary of all results."""
        result = {
            "match": {
                "home_team": home_team,
                "away_team": away_team,
                "date": match_date,
            },
            "features": features.to_dict(),
            "semantic_anchors": list(features.index),
        }

        if prediction is not None:
            # Assuming binary classification (H/D/A) – need class names
            # For simplicity, assume three classes: home, draw, away
            probs = prediction.flatten()
            if len(probs) == 3:
                result["prediction"] = {
                    "home_prob": float(probs[0]),
                    "draw_prob": float(probs[1]),
                    "away_prob": float(probs[2]),
                    "most_likely": ["H", "D", "A"][np.argmax(probs)],
                }
            elif len(probs) == 2:  # Binary, e.g., home win or not
                result["prediction"] = {
                    "home_win_prob": float(probs[1]),  # assuming class 1 is home win
                }
            else:
                result["prediction"] = {"probabilities": probs.tolist()}

        if actual is not None:
            result["actual"] = actual

        if validation is not None:
            result["validation"] = validation

        return result

    def _format_json(self, data: Dict[str, Any]) -> str:
        """Return JSON string with indentation."""
        return json.dumps(data, indent=2, default=str)

    def _format_human(self, data: Dict[str, Any]) -> str:
        """Return a human-readable text summary."""
        lines = []
        match = data["match"]
        lines.append(f"Match: {match['home_team']} vs {match['away_team']}")
        if match['date']:
            lines.append(f"Date: {match['date']}")
        lines.append("")

        # Top semantic anchors
        features = data["features"]
        # Sort by score descending, take top 5
        top_anchors = sorted(features.items(), key=lambda x: x[1], reverse=True)[:5]
        lines.append("Top 5 influencing factors:")
        for concept, score in top_anchors:
            lines.append(f"  {concept}: {score:.3f}")
        lines.append("")

        # Prediction
        if "prediction" in data:
            pred = data["prediction"]
            if "home_prob" in pred:
                lines.append(
                    f"Prediction: Home {pred['home_prob']:.1%} | Draw {pred['draw_prob']:.1%} | Away {pred['away_prob']:.1%}"
                )
                lines.append(f"Most likely: {pred['most_likely']}")
            elif "home_win_prob" in pred:
                lines.append(f"Home win probability: {pred['home_win_prob']:.1%}")
            else:
                lines.append(f"Prediction probabilities: {pred['probabilities']}")
            lines.append("")

        # Actual result
        if "actual" in data:
            lines.append(f"Actual result: {data['actual']}")
            if "validation" in data:
                val = data["validation"]
                if val.get("correct"):
                    lines.append("✓ Prediction was correct")
                else:
                    lines.append("✗ Prediction was incorrect")
        lines.append("")

        return "\n".join(lines)

    def _format_csv(self, data: Dict[str, Any]) -> str:
        """
        Return CSV string. For a single prediction, this produces one row with
        match info and all anchor scores as columns.
        """
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        header = ["home_team", "away_team", "date", "actual", "predicted_class", "home_prob", "draw_prob", "away_prob"]
        # Add anchor names
        anchor_names = list(data["features"].keys())
        header.extend(anchor_names)
        writer.writerow(header)

        # Row
        match = data["match"]
        row = [
            match["home_team"],
            match["away_team"],
            match["date"] or "",
            data.get("actual", ""),
        ]
        if "prediction" in data:
            pred = data["prediction"]
            if "most_likely" in pred:
                row.append(pred["most_likely"])
                row.append(pred.get("home_prob", ""))
                row.append(pred.get("draw_prob", ""))
                row.append(pred.get("away_prob", ""))
            else:
                row.extend(["", "", "", ""])
        else:
            row.extend(["", "", "", ""])

        # Add anchor scores
        for anchor in anchor_names:
            row.append(data["features"].get(anchor, ""))
        writer.writerow(row)

        return output.getvalue()

    def _format_markdown(self, data: Dict[str, Any]) -> str:
        """Return a Markdown summary, useful for reports."""
        lines = []
        match = data["match"]
        lines.append(f"# Match: {match['home_team']} vs {match['away_team']}")
        if match['date']:
            lines.append(f"**Date:** {match['date']}")
        lines.append("")

        # Feature table
        lines.append("## Semantic Anchor Scores")
        lines.append("| Anchor | Score |")
        lines.append("|--------|-------|")
        features = data["features"]
        # Sort by score descending
        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
        for concept, score in sorted_features:
            lines.append(f"| {concept} | {score:.3f} |")
        lines.append("")

        # Prediction
        if "prediction" in data:
            pred = data["prediction"]
            lines.append("## Prediction")
            if "home_prob" in pred:
                lines.append(f"- Home win: {pred['home_prob']:.1%}")
                lines.append(f"- Draw: {pred['draw_prob']:.1%}")
                lines.append(f"- Away win: {pred['away_prob']:.1%}")
                lines.append(f"- **Most likely:** {pred['most_likely']}")
            elif "home_win_prob" in pred:
                lines.append(f"- Home win probability: {pred['home_win_prob']:.1%}")
            else:
                lines.append(f"- Probabilities: {pred['probabilities']}")
            lines.append("")

        # Actual
        if "actual" in data:
            lines.append(f"## Actual Result: {data['actual']}")

        return "\n".join(lines)

    def _write_to_file(self, content: str) -> None:
        """Write formatted content to file."""
        try:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_file, "w") as f:
                f.write(content)
            logger.info(f"Output written to {self.output_file}")
        except Exception as e:
            logger.error(f"Failed to write output to {self.output_file}: {e}")
            raise

    def format_batch(self, results: List[Dict[str, Any]]) -> Union[str, None]:
        """
        Format a list of results (from batch processing) into a single output.
        For CSV, this produces multiple rows. For JSON, a list of objects.
        For human-readable, a concatenation of individual summaries.
        """
        if self.output_format == "csv":
            # For CSV, we need to combine all rows. Reuse _format_csv but for multiple results.
            import io

            output = io.StringIO()
            writer = csv.writer(output)

            # Determine header from first result
            if not results:
                return ""
            first = results[0]["formatted_output"] if "formatted_output" in results[0] else self._build_result_dict(
                results[0]["match_info"]["home_team"],
                results[0]["match_info"]["away_team"],
                results[0]["match_info"]["date"],
                results[0]["features"],
                results[0].get("prediction"),
                results[0].get("actual"),
                results[0].get("validation"),
            )
            # We can just call _format_csv on each and concatenate rows, but easier: build list of dicts and use pandas.
            # Simpler: use pandas to write CSV.
            import pandas as pd

            rows = []
            for r in results:
                match = r["match_info"]
                features = r["features"]
                pred = r.get("prediction")
                actual = r.get("actual")
                row = {
                    "home_team": match["home_team"],
                    "away_team": match["away_team"],
                    "date": match.get("date", ""),
                    "actual": actual or "",
                }
                if pred is not None:
                    # Assume 3-class probs
                    probs = pred.flatten() if isinstance(pred, np.ndarray) else pred
                    if len(probs) == 3:
                        row["predicted_class"] = ["H", "D", "A"][np.argmax(probs)]
                        row["home_prob"] = probs[0]
                        row["draw_prob"] = probs[1]
                        row["away_prob"] = probs[2]
                # Add anchor scores
                for anchor, score in features.items():
                    row[anchor] = score
                rows.append(row)

            df = pd.DataFrame(rows)
            return df.to_csv(index=False)

        elif self.output_format == "json":
            # Return list of result dicts
            return json.dumps([self._build_result_dict(
                r["match_info"]["home_team"],
                r["match_info"]["away_team"],
                r["match_info"]["date"],
                r["features"],
                r.get("prediction"),
                r.get("actual"),
                r.get("validation"),
            ) for r in results], indent=2, default=str)

        else:
            # For human-readable or markdown, concatenate with separators
            parts = []
            for r in results:
                # Recreate the formatted output for each result
                formatted = self._format_human(self._build_result_dict(
                    r["match_info"]["home_team"],
                    r["match_info"]["away_team"],
                    r["match_info"]["date"],
                    r["features"],
                    r.get("prediction"),
                    r.get("actual"),
                    r.get("validation"),
                ))
                parts.append(formatted)
                parts.append("-" * 50 + "\n")
            output = "\n".join(parts)
            if self.output_file:
                self._write_to_file(output)
                return None
            return output