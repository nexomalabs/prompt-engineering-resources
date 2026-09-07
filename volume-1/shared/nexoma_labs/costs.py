"""Token cost accounting.

Introduced in Volume I, Chapter 9 (Tokens, Context Windows, and Cost Economics)
and reused by every later lab that calls a model. Prices are supplied by the
caller rather than hard-coded, because published rates change and a book that
hard-codes them is wrong within months.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Price:
    """Rates in USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float = 0.0

    def __post_init__(self) -> None:
        for name in ("input_per_mtok", "output_per_mtok", "cached_input_per_mtok"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")


@dataclass
class Call:
    label: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost: float


@dataclass
class CostLedger:
    """Accumulates the cost of a sequence of model calls.

    The ledger is what turns "this seems expensive" into a number a reader can
    put in a design document.
    """

    price: Price
    calls: list = field(default_factory=list)

    def record(
        self,
        label: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        if cached_tokens > input_tokens:
            raise ValueError("cached_tokens cannot exceed input_tokens")
        for name, value in (
            ("input_tokens", input_tokens),
            ("output_tokens", output_tokens),
            ("cached_tokens", cached_tokens),
        ):
            if value < 0:
                raise ValueError(f"{name} must not be negative")

        fresh = input_tokens - cached_tokens
        cost = (
            fresh * self.price.input_per_mtok
            + cached_tokens * self.price.cached_input_per_mtok
            + output_tokens * self.price.output_per_mtok
        ) / 1_000_000
        self.calls.append(Call(label, input_tokens, output_tokens, cached_tokens, cost))
        return cost

    @property
    def total(self) -> float:
        return sum(c.cost for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.input_tokens + c.output_tokens for c in self.calls)

    def project(self, calls_per_day: int) -> float:
        """Daily cost if this workload ran `calls_per_day` times."""
        if not self.calls:
            return 0.0
        return self.total / len(self.calls) * calls_per_day

    def report(self) -> str:
        if not self.calls:
            return "no calls recorded"
        width = max(len(c.label) for c in self.calls)
        lines = [
            f"{'call'.ljust(width)}   {'in':>7} {'cached':>7} {'out':>7} {'USD':>10}",
            "-" * (width + 36),
        ]
        for c in self.calls:
            lines.append(
                f"{c.label.ljust(width)}   {c.input_tokens:>7} "
                f"{c.cached_tokens:>7} {c.output_tokens:>7} {c.cost:>10.6f}"
            )
        lines.append("-" * (width + 36))
        lines.append(f"{'TOTAL'.ljust(width)}   {'':>7} {'':>7} {'':>7} {self.total:>10.6f}")
        return "\n".join(lines)


def estimate(price: Price, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
    """One-shot cost of a single call, without a ledger."""
    return CostLedger(price).record("estimate", input_tokens, output_tokens, cached_tokens)
