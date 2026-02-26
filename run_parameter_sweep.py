import argparse
import csv
import os
import math
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict

from abm_simulation import SimulationConfig, MarketSimulation

def run_simulation(args) -> List[Dict]:
    """
    Runs a single simulation and returns key metrics per step.
    """
    run_id, self_share, steps, seed = args
    
    # Configure simulation
    config = SimulationConfig(
        time_steps=steps,
        self_share=self_share,
        seed=seed,
        # Use exogenous maturity to control comparison
        maturity_rule="exogenous",
        # Use a slightly higher growth rate to ensure we sweep 0.0 -> 1.0 within 200 steps (0.005 * 200 = 1.0)
        maturity_growth=0.005, 
        initial_buyers=400,
        
        # --- Tuned Parameters for Monotonic Threshold Decrease ---
        # 1. Base Penalty: Ensures crowding-out exists even at M=0 (prevents 1.0 threshold)
        base_self_penalty=0.15,         # New parameter
        
        # 2. Smooth Trust Jump: Avoids cliffs
        trust_jump_steepness=3.0,       # Gentle slope
        trust_jump_strength=0.2,        
        
        # 3. Early Bonus: High enough for "Significant Positive Effect", but not overwhelming
        early_entry_bonus=0.3,          # Boost demand level
        early_purchase_bonus=0.2,       
        
        # 4. Mature Penalty: Strong enough to push threshold left as M increases
        mature_self_penalty=0.8,        
        mature_congestion_penalty=0.4,  # Increased congestion impact
        
        # 5. Threshold Base
        threshold_base=0.45,            
        threshold_maturity_slope=0.4    
    )
    
    sim = MarketSimulation(config, seed)
    history = []
    
    for _ in range(steps):
        # Collect data at each step (which corresponds to a specific maturity level)
        step_data = sim.step()
        
        # We need to capture the relationship: self_share -> trade_volume | maturity
        history.append({
            "run_id": run_id,
            "self_share": self_share,
            "step": step_data["step"],
            "maturity": step_data["maturity"],
            "trade_volume": step_data["trade_volume"],
            "trust": step_data["trust"],
            "third_party_share": step_data["third_party_share"]
        })
        
    return history

def run_parameter_sweep(output_file: str, steps: int = 200, reps: int = 20):
    """
    Runs simulations for self_share from 0.0 to 1.0 with step 0.05.
    """
    # Finer grain sweep: 0.0, 0.05, ... 1.0
    self_shares = [round(x * 0.05, 2) for x in range(21)] 
    tasks = []
    
    print(f"Starting parameter sweep for self_shares: {self_shares}")
    print(f"Repetitions per setting: {reps}")
    
    task_id = 0
    for share in self_shares:
        for r in range(reps):
            seed = 42 + task_id # Different seed per run
            tasks.append((task_id, share, steps, seed))
            task_id += 1
            
    # Run in parallel
    results = []
    with ProcessPoolExecutor() as executor:
        for batch in executor.map(run_simulation, tasks):
            results.extend(batch)
            
    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["run_id", "self_share", "step", "maturity", "trade_volume", "trust", "third_party_share"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Sweep completed. Data saved to {output_file}")

if __name__ == "__main__":
    output_path = "experiment_output/parameter_sweep.csv"
    run_parameter_sweep(output_path, steps=200, reps=15)
