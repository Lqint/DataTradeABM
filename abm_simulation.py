from __future__ import annotations

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass
class SimulationConfig:
    time_steps: int = 200
    repetitions: int = 30
    seed: int = 42
    initial_providers: int = 80
    initial_buyers: int = 400
    market_growth_per_step: float = 1.0
    base_entry_rate: float = 5.0
    exit_threshold: float = -0.5
    exit_patience: int = 3
    self_share: float = 0.2
    governance_strength: float = 0.5
    matching_efficiency: float = 0.8
    demand_weight: float = 0.5
    buyer_entry_cost: float = 0.3
    self_quality: float = 0.8
    provider_quality_mean: float = 0.6
    provider_quality_std: float = 0.15
    self_price: float = 1.0
    provider_price: float = 1.0
    provider_price_premium: float = 0.08
    provider_variable_cost: float = 0.0
    provider_fixed_cost: float = 0.2
    provider_setup_cost: float = 2.0
    trust_base: float = 0.5
    trust_sensitivity: float = 0.5
    brand_compliance_base: float = 0.15
    brand_self_share_weight: float = 1.6
    brand_self_share_power: float = 2.2
    brand_trade_weight: float = 0.0
    brand_governance_weight: float = 0.2
    brand_entry_weight: float = 1.1
    brand_purchase_weight: float = 0.8
    maturity_rule: str = "endogenous"
    maturity_lambda: float = 0.08
    maturity_threshold: float = 120.0
    maturity_growth: float = 0.01
    info_mode: str = "full"
    info_noise_std: float = 0.08
    info_proxy_weight: float = 0.7
    learning_rate: float = 0.2
    enforce_separation_at: int | None = None
    separation_target: float = 0.0
    separation_maturity_threshold: float = 0.5
    network_effect: float = 0.05
    learning_effect: float = 0.1
    self_capacity_scale: float = 2.0
    purchase_sensitivity: float = 2.0
    purchase_trust_weight: float = 0.2
    entry_profit_sensitivity: float = 1.5
    self_exposure_bias: float = 0.6
    exposure_quality_boost: float = 0.2
    early_self_entry_relief: float = 0.95
    early_self_trust_boost: float = 0.7
    early_self_exposure_boost: float = 0.5
    early_provider_cushion: float = 0.35
    trust_jump_threshold: float = 0.35
    trust_jump_steepness: float = 10.0
    trust_jump_strength: float = 0.35
    trust_jump_floor: float = 0.08
    trust_jump_maturity_slope: float = 0.7
    threshold_base: float = 0.5
    threshold_maturity_slope: float = 0.45
    mature_trust_drag: float = 0.18
    mature_provider_penalty: float = 0.45
    mature_exit_threshold_boost: float = 0.3
    mature_exit_patience_reduction: int = 1
    mature_quality_invest_penalty: float = 0.2
    hard_jump_strength: float = 0.7
    hard_entry_waiver: float = 0.8
    hard_jump_maturity_power: float = 1.2
    early_trust_weight_boost: float = 0.5
    base_self_penalty: float = 0.1
    mature_self_penalty: float = 0.35
    mature_congestion_penalty: float = 0.2
    early_price_relief: float = 0.35
    early_entry_bonus: float = 0.4
    early_purchase_bonus: float = 0.3
    provider_visibility_base: float = 0.0
    separation_visibility_boost: float = 0.25
    separation_cost_reduction: float = 0.3
    separation_governance_boost: float = 0.15
    separation_info_noise_reduction: float = 0.05
    separation_proxy_boost: float = 0.1
    segment_count: int = 5
    segment_match_weight: float = 0.6
    match_preference_weight: float = 0.4
    self_segment_coverage: float = 0.7
    price_sensitivity: float = 1.0
    buyer_budget_mean: float = 1.1
    buyer_budget_std: float = 0.25
    buyer_budget_min: float = 0.4
    buyer_budget_max: float = 2.0
    budget_sensitivity: float = 3.0
    matching_weight: float = 0.3
    supply_weight: float = 0.2
    trade_unit: float = 1.0
    quality_weight: float = 1.2
    price_weight: float = 1.0
    self_quality_invest_rate: float = 0.06
    provider_quality_invest_rate: float = 0.04
    quality_diminish_power: float = 2.0


@dataclass
class Provider:
    quality: float
    variable_cost: float
    price: float
    segment: int
    setup_cost: float = 0.0
    setup_paid: bool = False
    loss_streak: int = 0


@dataclass
class Buyer:
    preference: float
    entry_cost: float
    belief_quality: float
    segment_preference: int
    budget: float


@dataclass
class PlatformState:
    maturity: float = 0.0
    trust: float = 0.5
    quality_signal: float = 0.5
    brand_compliance: float = 0.5
    last_orders: int = 0
    last_self_trade_intensity: float = 0.0


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def lhs_samples(n: int, param_ranges: Dict[str, Tuple[float, float]], seed: int) -> List[Dict[str, float]]:
    rng = random.Random(seed)
    keys = list(param_ranges.keys())
    bins = list(range(n))
    samples = [{k: 0.0 for k in keys} for _ in range(n)]
    for k in keys:
        rng.shuffle(bins)
        low, high = param_ranges[k]
        span = high - low
        for i, b in enumerate(bins):
            u = (b + rng.random()) / n
            samples[i][k] = low + span * u
    return samples


class MarketSimulation:
    def __init__(self, config: SimulationConfig, seed: int):
        self.config = config
        self.rng = random.Random(seed)
        self.base_self_share = config.self_share
        self.platform = PlatformState(
            maturity=0.0,
            trust=config.trust_base,
            quality_signal=0.5,
            brand_compliance=config.brand_compliance_base,
        )
        self.providers = self._init_providers()
        self.buyers = self._init_buyers()
        self.step_index = 0
        self.separation_start = None
        self.separation_active = False
        self.last_entry_rate = 0.0
        self.last_exit_rate = 0.0

    def _init_providers(self) -> List[Provider]:
        providers: List[Provider] = []
        for _ in range(self.config.initial_providers):
            quality = clamp(self.rng.gauss(self.config.provider_quality_mean, self.config.provider_quality_std), 0.0, 1.0)
            segment = self.rng.randrange(max(self.config.segment_count, 1))
            providers.append(
                Provider(
                    quality=quality,
                    variable_cost=self.config.provider_variable_cost,
                    price=self.config.provider_price,
                    segment=segment,
                    setup_cost=self.config.provider_setup_cost,
                )
            )
        return providers

    def _init_buyers(self) -> List[Buyer]:
        buyers: List[Buyer] = []
        for _ in range(self.config.initial_buyers):
            buyers.append(self._create_buyer())
        return buyers

    def _create_buyer(self) -> Buyer:
        preference = clamp(self.rng.random(), 0.2, 1.0)
        segment_preference = self.rng.randrange(max(self.config.segment_count, 1))
        budget = clamp(self.rng.gauss(self.config.buyer_budget_mean, self.config.buyer_budget_std), self.config.buyer_budget_min, self.config.buyer_budget_max)
        return Buyer(
            preference=preference,
            entry_cost=self.config.buyer_entry_cost,
            belief_quality=0.5,
            segment_preference=segment_preference,
            budget=budget,
        )

    def _expand_market(self) -> None:
        target = self.config.initial_buyers + int(self.config.market_growth_per_step * self.step_index)
        if target > len(self.buyers):
            for _ in range(target - len(self.buyers)):
                self.buyers.append(self._create_buyer())

    def _update_maturity(self, orders: int) -> None:
        if self.config.maturity_rule == "endogenous":
            x = self.config.maturity_lambda * (orders - self.config.maturity_threshold)
            self.platform.maturity = clamp(1.0 / (1.0 + math.exp(-x)), 0.0, 1.0)
        else:
            self.platform.maturity = clamp(self.platform.maturity + self.config.maturity_growth, 0.0, 1.0)

    def _maybe_apply_separation(self) -> None:
        if self.config.enforce_separation_at is not None:
            if self.step_index < self.config.enforce_separation_at:
                return
        else:
            if self.platform.maturity < self.config.separation_maturity_threshold:
                return
        if self.separation_start is None:
            self.separation_start = self.step_index
            self.separation_active = True
            self.config.governance_strength = clamp(
                self.config.governance_strength + self.config.separation_governance_boost,
                0.0,
                1.0,
            )
            self.config.info_noise_std = max(0.0, self.config.info_noise_std - self.config.separation_info_noise_reduction)
            self.config.info_proxy_weight = clamp(self.config.info_proxy_weight + self.config.separation_proxy_boost, 0.0, 1.0)
        self.config.self_share = self.config.separation_target

    def _quality_signal(self) -> float:
        provider_quality = sum(p.quality for p in self.providers) / max(len(self.providers), 1)
        q_pool = self.config.self_share * self.config.self_quality + (1 - self.config.self_share) * provider_quality
        if self.config.info_mode == "full":
            return q_pool
        if self.config.info_mode == "noisy":
            noisy = q_pool + self.rng.gauss(0.0, self.config.info_noise_std)
            return clamp(noisy, 0.0, 1.0)
        proxy = clamp(self.platform.trust, 0.0, 1.0)
        return clamp(self.config.info_proxy_weight * q_pool + (1 - self.config.info_proxy_weight) * proxy, 0.0, 1.0)

    def _update_trust(self, quality_signal: float) -> None:
        governance_shift = 0.2 * (self.config.governance_strength - 0.5)
        maturity = self.platform.maturity
        maturity_factor = max(0.0, 1.0 - maturity)
        early_factor = math.sqrt(self.config.self_share) * maturity_factor
        threshold = clamp(
            self.config.threshold_base - self.config.threshold_maturity_slope * maturity,
            0.0,
            1.0,
        )
        jump_input = self.config.self_share - max(threshold, self.config.trust_jump_threshold)
        jump_core = logistic(self.config.trust_jump_steepness * jump_input)
        jump_decay = self.config.trust_jump_floor + (1.0 - self.config.trust_jump_floor) * (maturity_factor ** self.config.trust_jump_maturity_slope)
        trust_jump = self.config.trust_jump_strength * jump_core * jump_decay
        target = clamp(
            self.config.trust_base
            + self.config.trust_sensitivity * (quality_signal - 0.5)
            + governance_shift
            + self.config.early_self_trust_boost * early_factor
            + trust_jump
            - self.config.mature_trust_drag * maturity * self.config.self_share,
            0.0,
            1.0,
        )
        self.platform.trust = clamp(self.platform.trust + self.config.learning_rate * (target - self.platform.trust), 0.0, 1.0)

    def _update_brand_compliance(self) -> None:
        share_term = self.base_self_share ** self.config.brand_self_share_power
        score = (
            self.config.brand_compliance_base
            + self.config.brand_self_share_weight * share_term
            + self.config.brand_trade_weight * self.platform.last_self_trade_intensity
            + self.config.brand_governance_weight * self.config.governance_strength
        )
        self.platform.brand_compliance = clamp(score, 0.0, 1.0)

    def _buyer_entry(self, buyer: Buyer, perceived_quality: float, price: float) -> bool:
        maturity = self.platform.maturity
        maturity_factor = max(0.0, 1.0 - maturity)
        early_factor = math.sqrt(self.config.self_share) * maturity_factor
        threshold = clamp(
            self.config.threshold_base - self.config.threshold_maturity_slope * maturity,
            0.0,
            1.0,
        )
        # Soft sigmoid for hard_jump
        hard_jump = logistic(10.0 * (self.config.self_share - threshold))
        effective_entry_cost = buyer.entry_cost * (1.0 - 0.3 * self.config.governance_strength)
        # Saturated relief
        relief_factor = 1.0 - math.exp(-2.0 * self.config.self_share)
        effective_entry_cost = max(0.0, effective_entry_cost * (1.0 - self.config.early_self_entry_relief * relief_factor * maturity_factor))
        if hard_jump > 0.0:
            hard_factor = (maturity_factor ** self.config.hard_jump_maturity_power)
            effective_entry_cost = max(0.0, effective_entry_cost * (1.0 - self.config.hard_entry_waiver * hard_factor))
        scale_effect = self.platform.last_orders / max(len(self.buyers), 1)
        expected_utility = (
            buyer.preference * perceived_quality
            - price
            - effective_entry_cost
            + self.config.brand_entry_weight * self.platform.brand_compliance
            + hard_jump * self.config.early_entry_bonus * maturity_factor
        )
        probability = logistic(expected_utility + self.config.network_effect * scale_effect + 0.2 * self.config.governance_strength)
        return self.rng.random() < probability

    def _buyer_purchase(
        self,
        buyer: Buyer,
        effective_quality: float,
        price: float,
        match_efficiency: float,
        supply_ratio: float,
    ) -> bool:
        maturity = self.platform.maturity
        maturity_factor = max(0.0, 1.0 - maturity)
        early_factor = math.sqrt(self.config.self_share) * maturity_factor
        threshold = clamp(
            self.config.threshold_base - self.config.threshold_maturity_slope * maturity,
            0.0,
            1.0,
        )
        # Soft sigmoid for hard_jump
        hard_jump = logistic(10.0 * (self.config.self_share - threshold))
        effective_trust_weight = self.config.purchase_trust_weight + self.config.early_trust_weight_boost * early_factor
        effective_price_sensitivity = self.config.price_sensitivity * (1.0 - self.config.early_price_relief * early_factor)
        
        # Enhanced non-linear penalty for mature self-share
        # Added base_self_penalty to ensure some crowding-out exists even at low maturity
        total_penalty_coeff = self.config.base_self_penalty + self.config.mature_self_penalty * maturity
        mature_penalty_term = total_penalty_coeff * (self.config.self_share ** 2)
        
        expected_utility = (
            buyer.preference * effective_quality
            - effective_price_sensitivity * price
            + effective_trust_weight * self.platform.trust
            + self.config.matching_weight * (match_efficiency - 0.5)
            + self.config.supply_weight * (supply_ratio - 0.5)
            + self.config.brand_purchase_weight * self.platform.brand_compliance
            + hard_jump * self.config.early_purchase_bonus * maturity_factor
            - mature_penalty_term
        )
        budget_margin = logistic(self.config.budget_sensitivity * (buyer.budget - price))
        probability = logistic(self.config.purchase_sensitivity * expected_utility) * budget_margin
        return self.rng.random() < probability

    def _pick_supply(self, avg_provider_quality: float) -> str:
        provider_visibility = 1.0 + self.config.provider_visibility_base + 0.3 * self.config.governance_strength
        self_visibility = 1.0 + self.config.self_exposure_bias
        if self.separation_active:
            provider_visibility += self.config.separation_visibility_boost
            self_visibility = max(0.0, self_visibility - self.config.separation_visibility_boost)
        provider_score = math.exp(
            self.config.quality_weight * avg_provider_quality
            - self.config.price_weight * (self.config.provider_price + self.config.provider_price_premium)
        ) * provider_visibility
        self_score = math.exp(
            self.config.quality_weight * self.config.self_quality - self.config.price_weight * self.config.self_price
        ) * self_visibility
        total = self_score + provider_score
        if total <= 0:
            return "provider"
        return "self" if self.rng.random() < (self_score / total) else "provider"

    def _provider_match(self, provider: Provider, buyer: Buyer) -> float:
        if self.config.segment_count <= 1:
            return 1.0
        distance = abs(provider.segment - buyer.segment_preference)
        return 1.0 - (distance / max(self.config.segment_count - 1, 1))

    def _select_provider(self, buyer: Buyer) -> int:
        weights: List[float] = []
        for provider in self.providers:
            match_score = self._provider_match(provider, buyer)
            score = math.exp(
                self.config.quality_weight * provider.quality
                + self.config.segment_match_weight * match_score
                - self.config.price_weight * (provider.price + self.config.provider_price_premium)
            )
            weights.append(score)
        total = sum(weights)
        if total <= 0:
            return self.rng.randrange(len(self.providers))
        pick = self.rng.random() * total
        cumulative = 0.0
        for idx, weight in enumerate(weights):
            cumulative += weight
            if pick <= cumulative:
                return idx
        return len(weights) - 1

    def _update_provider_population(self, trades_by_provider: List[int]) -> None:
        next_providers: List[Provider] = []
        prev_count = max(len(self.providers), 1)
        maturity = self.platform.maturity
        early_factor = math.sqrt(self.config.self_share) * max(0.0, 1.0 - maturity)
        cost_multiplier = 1.0 - (self.config.separation_cost_reduction if self.separation_active else 0.0)
        cost_multiplier = max(0.0, cost_multiplier * (1.0 - self.config.early_provider_cushion * early_factor))
        # Non-linear penalty for mature factor
        mature_factor = maturity * (self.config.self_share ** 2)
        effective_fixed_cost = self.config.provider_fixed_cost * cost_multiplier
        for provider, trades in zip(self.providers, trades_by_provider):
            trade_intensity = trades / max(self.platform.last_orders, 1)
            invest_strength = self.config.provider_quality_invest_rate * trade_intensity * (1.0 - self.config.mature_quality_invest_penalty * mature_factor)
            delta_quality = invest_strength * ((1.0 - provider.quality) ** self.config.quality_diminish_power)
            provider.quality = clamp(provider.quality + delta_quality, 0.0, 1.0)
            setup_charge = (provider.setup_cost * cost_multiplier) if not provider.setup_paid else 0.0
            profit = trades * (provider.price - provider.variable_cost) - effective_fixed_cost - setup_charge
            provider.setup_paid = True
            exit_threshold = self.config.exit_threshold + self.config.mature_exit_threshold_boost * mature_factor
            if profit < exit_threshold:
                provider.loss_streak += 1
            else:
                provider.loss_streak = 0
            exit_patience = max(1, self.config.exit_patience - int(self.config.mature_exit_patience_reduction * mature_factor))
            if provider.loss_streak < exit_patience:
                next_providers.append(provider)
        survivors = len(next_providers)
        exit_count = max(prev_count - survivors, 0)
        provider_visibility = 1.0 + self.config.provider_visibility_base + 0.3 * self.config.governance_strength
        if self.separation_active:
            provider_visibility += self.config.separation_visibility_boost
        entry_multiplier = (0.6 + 0.8 * self.config.governance_strength) * max(self.platform.trust, 0.1) * provider_visibility
        entry_multiplier = max(0.0, entry_multiplier * (1.0 - self.config.mature_provider_penalty * mature_factor))
        avg_trades = sum(trades_by_provider) / max(len(trades_by_provider), 1) if trades_by_provider else 0.0
        demand_per_provider = max(self.platform.last_orders, 1) / max(len(self.providers), 1)
        expected_trades_per_provider = 0.6 * avg_trades + 0.4 * demand_per_provider * self.config.matching_efficiency
        effective_provider_price = self.config.provider_price + self.config.provider_price_premium
        expected_profit = expected_trades_per_provider * (effective_provider_price - self.config.provider_variable_cost) - effective_fixed_cost - (self.config.provider_setup_cost * cost_multiplier)
        entry_signal = logistic(self.config.entry_profit_sensitivity * expected_profit)
        entry_count = int(self.config.base_entry_rate * entry_multiplier * entry_signal)
        for _ in range(entry_count):
            quality = clamp(self.rng.gauss(self.config.provider_quality_mean, self.config.provider_quality_std), 0.0, 1.0)
            segment = self.rng.randrange(max(self.config.segment_count, 1))
            next_providers.append(
                Provider(
                    quality=quality,
                    variable_cost=self.config.provider_variable_cost,
                    price=self.config.provider_price,
                    segment=segment,
                    setup_cost=self.config.provider_setup_cost,
                )
            )
        self.providers = next_providers
        self.last_entry_rate = entry_count / max(prev_count, 1)
        self.last_exit_rate = exit_count / max(prev_count, 1)

    def step(self) -> Dict[str, float]:
        self._expand_market()
        quality_signal = self._quality_signal()
        self.platform.quality_signal = quality_signal
        self._update_trust(quality_signal)
        self._update_brand_compliance()

        orders = 0
        trades = 0
        self_trades = 0
        provider_trades = [0 for _ in self.providers]
        provider_trade_matches = 0.0
        provider_purchase_count = 0

        average_price = (self.config.self_share * self.config.self_price) + ((1 - self.config.self_share) * (self.config.provider_price + self.config.provider_price_premium))
        avg_provider_quality = sum(p.quality for p in self.providers) / max(len(self.providers), 1)
        # Modified match efficiency to include base congestion penalty
        # Congestion factor scales from 0.2 (base) to 1.0 (mature)
        congestion_factor = 0.2 + 0.8 * self.platform.maturity
        match_efficiency = self.config.matching_efficiency + 0.2 * self.platform.trust - self.config.mature_congestion_penalty * congestion_factor * self.config.self_share
        match_efficiency = clamp(match_efficiency, 0.0, 1.0)
        self_capacity = self.config.self_capacity_scale * self.config.self_share * self.config.initial_providers * (1.0 + self.platform.maturity)
        supply_ratio = (max(len(self.providers), 1) + self_capacity) / (self.config.initial_providers + self_capacity)
        for buyer in self.buyers:
            perceived = buyer.belief_quality
            if self.config.info_mode == "full":
                perceived = quality_signal
            elif self.config.info_mode == "noisy":
                perceived = clamp(quality_signal + self.rng.gauss(0.0, self.config.info_noise_std), 0.0, 1.0)
            else:
                perceived = clamp(self.config.info_proxy_weight * quality_signal + (1 - self.config.info_proxy_weight) * buyer.belief_quality, 0.0, 1.0)
            early_factor = math.sqrt(self.config.self_share) * max(0.0, 1.0 - self.platform.maturity)
            exposure_boost = (
                self.config.exposure_quality_boost * math.tanh(2.0 * self.config.self_share) * self.config.self_exposure_bias
                + self.config.early_self_exposure_boost * early_factor
            )
            perceived = clamp(perceived + exposure_boost, 0.0, 1.0)
            effective_learning = clamp(
                self.config.learning_rate + self.config.learning_effect * (self.platform.trust - 0.5),
                0.0,
                1.0,
            )
            buyer.belief_quality = clamp(buyer.belief_quality + effective_learning * (perceived - buyer.belief_quality), 0.0, 1.0)
            if self._buyer_entry(buyer, perceived, average_price):
                orders += 1
                choice = self._pick_supply(avg_provider_quality)
                if choice == "self":
                    chosen_price = self.config.self_price
                    effective_quality = clamp(
                        self.config.match_preference_weight * perceived
                        + (1 - self.config.match_preference_weight) * self.config.self_quality,
                        0.0,
                        1.0,
                    )
                    purchased = self._buyer_purchase(buyer, effective_quality, chosen_price, match_efficiency, supply_ratio)
                else:
                    chosen_price = self.config.provider_price + self.config.provider_price_premium
                    if self.providers:
                        idx = self._select_provider(buyer)
                        provider = self.providers[idx]
                        match_score = self._provider_match(provider, buyer)
                        effective_quality = clamp(
                            self.config.match_preference_weight * perceived
                            + (1 - self.config.match_preference_weight) * (0.6 * provider.quality + 0.4 * match_score),
                            0.0,
                            1.0,
                        )
                        purchased = self._buyer_purchase(buyer, effective_quality, chosen_price, match_efficiency, supply_ratio)
                    else:
                        purchased = self._buyer_purchase(buyer, perceived, chosen_price, match_efficiency, supply_ratio)
                if purchased:
                    trades += 1
                    if choice == "self":
                        self_trades += 1
                    else:
                        if provider_trades:
                            idx = self._select_provider(buyer)
                            provider_trades[idx] += 1
                            provider_trade_matches += match_score
                            provider_purchase_count += 1
                        else:
                            self_trades += 1

        self._update_provider_population(provider_trades)
        self._update_maturity(orders)
        self._maybe_apply_separation()
        self.platform.last_orders = orders

        avg_lot_size = self.config.trade_unit * (
            1.0
            + 0.6 * (self.platform.quality_signal - 0.5)
            + 0.4 * (self.platform.trust - 0.5)
        )
        avg_lot_size = max(0.1, avg_lot_size) * (1.0 + 0.2 * match_efficiency)
        trade_volume = trades * avg_lot_size
        third_party_trades = sum(provider_trades)
        third_party_share = (third_party_trades / trades) if trades > 0 else 0.0
        hhi = 0.0
        if third_party_trades > 0:
            shares = [t / third_party_trades for t in provider_trades if t > 0]
            hhi = sum(s * s for s in shares)
        match_rate = (provider_trade_matches / max(provider_purchase_count, 1)) if provider_purchase_count > 0 else 0.0
        self_trade_intensity = self_trades / max(orders, 1)
        self.platform.last_self_trade_intensity = self_trade_intensity
        self_invest = self.config.self_quality_invest_rate * self_trade_intensity
        self_delta = self_invest * ((1.0 - self.config.self_quality) ** self.config.quality_diminish_power)
        self.config.self_quality = clamp(self.config.self_quality + self_delta, 0.0, 1.0)

        self.step_index += 1
        return {
            "step": self.step_index,
            "orders": orders,
            "trades": trades,
            "self_trades": self_trades,
            "providers": len(self.providers),
            "quality_signal": quality_signal,
            "trust": self.platform.trust,
            "brand_compliance": self.platform.brand_compliance,
            "maturity": self.platform.maturity,
            "trade_volume": trade_volume,
            "entry_rate": self.last_entry_rate,
            "exit_rate": self.last_exit_rate,
            "provider_hhi": hhi,
            "third_party_share": third_party_share,
            "match_rate": match_rate,
        }


def run_experiment(config: SimulationConfig, run_id: int, seed: int) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    sim = MarketSimulation(config, seed)
    history: List[Dict[str, float]] = []
    for _ in range(config.time_steps):
        history.append(sim.step())
    avg_orders = sum(r["orders"] for r in history) / len(history)
    avg_trade = sum(r["trade_volume"] for r in history) / len(history)
    avg_providers = sum(r["providers"] for r in history) / len(history)
    avg_quality = sum(r["quality_signal"] for r in history) / len(history)
    avg_trust = sum(r["trust"] for r in history) / len(history)
    avg_brand_compliance = sum(r["brand_compliance"] for r in history) / len(history)
    avg_maturity = sum(r["maturity"] for r in history) / len(history)
    avg_entry_rate = sum(r.get("entry_rate", 0.0) for r in history) / len(history)
    avg_exit_rate = sum(r.get("exit_rate", 0.0) for r in history) / len(history)
    avg_hhi = sum(r.get("provider_hhi", 0.0) for r in history) / len(history)
    avg_third_party_share = sum(r.get("third_party_share", 0.0) for r in history) / len(history)
    avg_match_rate = sum(r.get("match_rate", 0.0) for r in history) / len(history)
    return history, {
        "run_id": run_id,
        "seed": seed,
        "avg_orders": avg_orders,
        "avg_trade_volume": avg_trade,
        "avg_providers": avg_providers,
        "avg_quality_signal": avg_quality,
        "avg_trust": avg_trust,
        "avg_brand_compliance": avg_brand_compliance,
        "avg_maturity": avg_maturity,
        "avg_entry_rate": avg_entry_rate,
        "avg_exit_rate": avg_exit_rate,
        "avg_provider_hhi": avg_hhi,
        "avg_third_party_share": avg_third_party_share,
        "avg_match_rate": avg_match_rate,
    }


def build_default_groups(base: SimulationConfig) -> Dict[str, SimulationConfig]:
    groups: Dict[str, SimulationConfig] = {}
    g1 = SimulationConfig(**base.__dict__)
    g1.self_share = 0.5
    g1.maturity_rule = "exogenous"
    g1.maturity_growth = 0.0025
    groups["G1_initial_high_self"] = g1

    g2 = SimulationConfig(**base.__dict__)
    g2.self_share = 0.5
    g2.maturity_rule = "exogenous"
    g2.maturity_growth = 0.02
    groups["G2_mature_high_self"] = g2

    g3 = SimulationConfig(**base.__dict__)
    g3.self_share = 0.5
    g3.separation_target = 0.0
    groups["G3_separation"] = g3

    c1 = SimulationConfig(**base.__dict__)
    c1.self_share = 0.1
    groups["C1_low_self"] = c1

    c1e = SimulationConfig(**base.__dict__)
    c1e.self_share = 0.1
    c1e.maturity_rule = "exogenous"
    c1e.maturity_growth = 0.0025
    groups["C1_early_low_self"] = c1e

    c2 = SimulationConfig(**base.__dict__)
    c2.governance_strength = 0.5
    groups["C2_neutral_governance"] = c2

    g4 = SimulationConfig(**base.__dict__)
    g4.network_effect = 0.15
    groups["G4_network_effect"] = g4

    g5 = SimulationConfig(**base.__dict__)
    g5.learning_rate = 0.4
    groups["G5_fast_learning"] = g5

    g6 = SimulationConfig(**base.__dict__)
    g6.info_mode = "noisy"
    g6.info_noise_std = 0.2
    groups["G6_info_asymmetry"] = g6

    return groups


def write_history(path: str, rows: Iterable[Dict[str, float]]) -> None:
    rows_list = list(rows)
    if not rows_list:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_list[0].keys()))
        writer.writeheader()
        writer.writerows(rows_list)


def write_summary(path: str, rows: Iterable[Dict[str, float]]) -> None:
    rows_list = list(rows)
    if not rows_list:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_list[0].keys()))
        writer.writeheader()
        writer.writerows(rows_list)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output", help="output directory")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lhs", type=int, default=0)
    parser.add_argument("--self-quality-grid", default="")
    args = parser.parse_args()

    base = SimulationConfig(time_steps=args.steps, repetitions=args.reps, seed=args.seed)
    os.makedirs(args.out, exist_ok=True)

    if args.lhs > 0:
        param_ranges = {
            "self_share": (0.0, 0.6),
            "governance_strength": (0.0, 1.0),
            "matching_efficiency": (0.4, 1.2),
            "buyer_entry_cost": (0.1, 0.6),
            "self_quality": (0.3, 0.8),
            "provider_quality_mean": (0.2, 0.95),
        }
        samples = lhs_samples(args.lhs, param_ranges, args.seed)
        summary_rows: List[Dict[str, float]] = []
        for i, sample in enumerate(samples, start=1):
            config = SimulationConfig(**base.__dict__)
            for k, v in sample.items():
                setattr(config, k, v)
            history, summary = run_experiment(config, i, args.seed + i)
            write_history(os.path.join(args.out, f"lhs_{i}_history.csv"), history)
            summary_rows.append(summary)
        write_summary(os.path.join(args.out, "lhs_summary.csv"), summary_rows)
        return

    if args.self_quality_grid:
        quality_values = [float(v.strip()) for v in args.self_quality_grid.split(",") if v.strip()]
        groups: Dict[str, SimulationConfig] = {}
        for value in quality_values:
            g1 = SimulationConfig(**base.__dict__)
            g1.self_share = 0.5
            g1.self_quality = value
            groups[f"SQ{value}_G1_initial_high_self"] = g1
            c1 = SimulationConfig(**base.__dict__)
            c1.self_share = 0.1
            c1.self_quality = value
            groups[f"SQ{value}_C1_low_self"] = c1
    else:
        groups = build_default_groups(base)
    summary_rows: List[Dict[str, float]] = []
    run_id = 0
    for group_name, config in groups.items():
        for rep in range(config.repetitions):
            run_id += 1
            history, summary = run_experiment(config, run_id, args.seed + run_id)
            summary["group"] = group_name
            write_history(os.path.join(args.out, f"{group_name}_{run_id}_history.csv"), history)
            summary_rows.append(summary)
    write_summary(os.path.join(args.out, "summary.csv"), summary_rows)


if __name__ == "__main__":
    main()
