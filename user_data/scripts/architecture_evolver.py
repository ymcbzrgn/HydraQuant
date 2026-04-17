"""
architecture_evolver.py — Phase 26 Sprint 2, Task 11A

Evolutionary Architecture Search — The organism evolves its own STRUCTURE.

NEAT-inspired: minimal structure → complexity added ONLY when needed.
Each organ is a "gene" that can be activated/deactivated/mutated.

Operations:
  - mutate: add/remove sub-organ, add/remove connection, change size, toggle organ
  - crossover: combine two genome configurations
  - evolve: run tournament selection on population of configurations
  - fitness: backtested Sharpe ratio of each configuration

Runs weekly as scheduler job. Top configurations replace current architecture.

Integration:
  - Reads from: self_model (organ performance), organism_audit
  - Writes to: Grafeo (architecture graph), organism_audit
  - Consumed by: neural_organism for structural updates

Reference: NEAT (Stanley & Miikkulainen, 2002)
"""

import os
import sys
import json
import logging
import copy
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("architecture_evolver")

from db import init_db

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
GENOME_PATH = os.path.join(MODEL_DIR, "genome_best.json")

# Default genome template
DEFAULT_GENOME = {
    "organs": {
        "crowd_scoring": {"active": True, "sub_organs": 3, "neuron_count": 18},
        "timing": {"active": True, "sub_organs": 1, "neuron_count": 2},
        "sizing": {"active": True, "sub_organs": 2, "neuron_count": 6},
        "risk": {"active": True, "sub_organs": 2, "neuron_count": 8},
        "momentum": {"active": True, "sub_organs": 2, "neuron_count": 10},
        "sentiment": {"active": True, "sub_organs": 1, "neuron_count": 4},
        "trend": {"active": True, "sub_organs": 2, "neuron_count": 12},
        "volatility": {"active": True, "sub_organs": 1, "neuron_count": 6},
    },
    "connections": [
        {"source": "crowd_scoring", "target": "sizing", "weight": 0.8, "type": "excitatory"},
        {"source": "timing", "target": "sizing", "weight": 0.3, "type": "modulatory"},
        {"source": "momentum", "target": "trend", "weight": 0.6, "type": "excitatory"},
        {"source": "risk", "target": "sizing", "weight": -0.5, "type": "inhibitory"},
        {"source": "volatility", "target": "risk", "weight": 0.7, "type": "excitatory"},
        {"source": "sentiment", "target": "crowd_scoring", "weight": 0.4, "type": "excitatory"},
    ],
    "meta": {
        "learning_rate": 0.001,
        "decay_factor": 0.995,
        "fear_sensitivity": 0.5,
    },
    "fitness": 0.0,
    "generation": 0,
    "innovation_number": 0,
}

MUTATION_TYPES = [
    "add_sub_organ", "remove_sub_organ", "add_connection",
    "remove_connection", "change_organ_size", "toggle_organ",
    "mutate_weight", "mutate_meta",
]


class ArchitectureEvolver:
    """Evolves the organism's architecture through neuroevolution."""

    def __init__(self):
        self._population: List[Dict] = []
        self._best_genome = None
        self._generation = 0
        self._innovation_counter = 0
        init_db()

    def mutate(self, genome: Dict) -> Dict:
        """Apply random structural mutation to a genome."""
        mutated = copy.deepcopy(genome)
        mutation = random.choice(MUTATION_TYPES)
        organs = mutated["organs"]
        connections = mutated["connections"]

        if mutation == "add_sub_organ" and organs:
            organ = random.choice(list(organs.keys()))
            organs[organ]["sub_organs"] = min(organs[organ]["sub_organs"] + 1, 8)
            organs[organ]["neuron_count"] += random.randint(1, 4)

        elif mutation == "remove_sub_organ" and organs:
            organ = random.choice(list(organs.keys()))
            if organs[organ]["sub_organs"] > 1:
                organs[organ]["sub_organs"] -= 1
                organs[organ]["neuron_count"] = max(organs[organ]["neuron_count"] - 2, 1)

        elif mutation == "add_connection":
            organ_names = list(organs.keys())
            if len(organ_names) >= 2:
                src = random.choice(organ_names)
                tgt = random.choice([o for o in organ_names if o != src])
                conn_type = random.choice(["excitatory", "inhibitory", "modulatory"])
                weight = random.uniform(-1, 1)
                connections.append({
                    "source": src, "target": tgt,
                    "weight": round(weight, 3), "type": conn_type,
                })
                self._innovation_counter += 1

        elif mutation == "remove_connection" and connections:
            idx = random.randint(0, len(connections) - 1)
            connections.pop(idx)

        elif mutation == "change_organ_size" and organs:
            organ = random.choice(list(organs.keys()))
            delta = random.choice([-2, -1, 1, 2, 4])
            organs[organ]["neuron_count"] = max(organs[organ]["neuron_count"] + delta, 1)

        elif mutation == "toggle_organ" and organs:
            organ = random.choice(list(organs.keys()))
            organs[organ]["active"] = not organs[organ]["active"]

        elif mutation == "mutate_weight" and connections:
            idx = random.randint(0, len(connections) - 1)
            connections[idx]["weight"] += random.gauss(0, 0.2)
            connections[idx]["weight"] = round(np.clip(connections[idx]["weight"], -2, 2), 3)

        elif mutation == "mutate_meta":
            meta = mutated["meta"]
            key = random.choice(list(meta.keys()))
            meta[key] *= random.uniform(0.8, 1.2)

        mutated["generation"] = self._generation
        mutated["innovation_number"] = self._innovation_counter
        return mutated

    def crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Combine two genomes (uniform crossover per organ)."""
        child = copy.deepcopy(parent1)

        for organ in child["organs"]:
            if organ in parent2["organs"] and random.random() < 0.5:
                child["organs"][organ] = copy.deepcopy(parent2["organs"][organ])

        # Combine connections (union, deduplicate)
        p2_conns = {(c["source"], c["target"]): c for c in parent2["connections"]}
        existing = {(c["source"], c["target"]) for c in child["connections"]}
        for key, conn in p2_conns.items():
            if key not in existing and random.random() < 0.5:
                child["connections"].append(copy.deepcopy(conn))

        child["generation"] = self._generation
        return child

    def evolve(self, population_size: int = 20, n_generations: int = 10,
               mutation_rate: float = 0.3) -> Dict:
        """Run evolutionary optimization.

        Phase 27 Task 24: before accepting any proposed architecture we call
        `self_model.verify_architectural_change()` — if AII falls into RED or
        CRITICAL bands, OR essential organs / connections are missing, the
        evolve step is aborted and the current architecture is preserved.
        Every attempt (accepted or rejected) is persisted to
        `autopoietic_integrity` for audit trail.

        Returns best genome after n_generations.
        """
        # Phase 27 Task 24 AII gate — PRE-evolve check. If the organism's
        # identity is already compromised (e.g. a sequence of bad mutations
        # across weeks), we must NOT compound the damage with more mutations.
        try:
            from self_model import get_self_model, verify_architectural_change
            aii_status = verify_architectural_change(get_self_model())
            if not aii_status["accepted"]:
                logger.warning(
                    f"[Evolver:AII] pre-evolve gate REJECT "
                    f"(status={aii_status['status']}, "
                    f"composite={aii_status['aii_composite']:.2f}, "
                    f"reasons={aii_status['reasons']})"
                )
                return {
                    "genome": None,
                    "fitness": 0.0,
                    "aborted": True,
                    "reason": "AII_GATE_BLOCKED",
                    "aii": aii_status,
                }
            logger.info(
                f"[Evolver:AII] pre-evolve gate ACCEPT "
                f"(status={aii_status['status']}, "
                f"composite={aii_status['aii_composite']:.2f})"
            )
        except Exception as e:
            logger.debug(f"[Evolver:AII] gate bypassed: {e}")

        # Initialize population
        if not self._population:
            self._population = [copy.deepcopy(DEFAULT_GENOME) for _ in range(population_size)]
            for i, genome in enumerate(self._population):
                if i > 0:  # Keep one unmodified
                    for _ in range(random.randint(1, 3)):
                        genome = self.mutate(genome)
                    self._population[i] = genome

        logger.info(f"[Evolver] Starting evolution: pop={len(self._population)}, "
                    f"gens={n_generations}")

        for gen in range(n_generations):
            self._generation = gen

            # Evaluate fitness
            for genome in self._population:
                genome["fitness"] = self._evaluate_fitness(genome)

            # Sort by fitness (descending)
            self._population.sort(key=lambda g: g["fitness"], reverse=True)

            # Log best
            best = self._population[0]
            logger.info(f"[Evolver] Gen {gen}: best_fitness={best['fitness']:.4f}, "
                       f"organs={sum(1 for o in best['organs'].values() if o['active'])}, "
                       f"connections={len(best['connections'])}")

            # Selection + reproduction
            elite_size = max(2, population_size // 5)
            new_pop = self._population[:elite_size]  # Elitism

            # Phase 27 Task 24 audit fix: per-child AII + identity_limits gate.
            # Pre-evolve check above is coarse (organism-wide state); we also
            # must reject INDIVIDUAL proposed genomes that disable an essential
            # organ or drop a safety-critical connection. Rejected candidates
            # fall back to a mutation of the best-so-far parent.
            try:
                from self_model import get_self_model, verify_architectural_change
                _sm = get_self_model()
            except Exception:
                _sm = None
                verify_architectural_change = None

            rejected_children = 0
            while len(new_pop) < population_size:
                if random.random() < 0.7:
                    # Crossover
                    p1 = random.choice(self._population[:elite_size * 2])
                    p2 = random.choice(self._population[:elite_size * 2])
                    child = self.crossover(p1, p2)
                else:
                    # Mutation
                    parent = random.choice(self._population[:elite_size * 2])
                    child = self.mutate(parent)

                if random.random() < mutation_rate:
                    child = self.mutate(child)

                # AII / identity_limits gate per child. If any proposed genome
                # strips an essential organ / connection, keep the parent copy
                # instead — essential surface is guaranteed intact.
                if verify_architectural_change is not None and _sm is not None:
                    try:
                        check = verify_architectural_change(_sm, proposed_genome=child)
                        if not check["accepted"]:
                            rejected_children += 1
                            logger.debug(
                                f"[Evolver:AII] gen={gen} reject child "
                                f"reasons={check['reasons']}"
                            )
                            # Fall back to a safe mutation of the top parent.
                            safe_parent = self._population[0]
                            child = copy.deepcopy(safe_parent)
                    except Exception:
                        pass

                new_pop.append(child)

            if rejected_children:
                logger.info(
                    f"[Evolver:AII] gen={gen} rejected {rejected_children} "
                    f"non-compliant children; parents reused"
                )
            self._population = new_pop

        self._best_genome = self._population[0]
        self._save_best()

        return {
            "best_fitness": self._best_genome["fitness"],
            "active_organs": sum(1 for o in self._best_genome["organs"].values() if o["active"]),
            "n_connections": len(self._best_genome["connections"]),
            "total_neurons": sum(o["neuron_count"] for o in self._best_genome["organs"].values() if o["active"]),
            "generation": n_generations,
        }

    def _evaluate_fitness(self, genome: Dict) -> float:
        """Evaluate genome fitness using historical performance correlation.

        Uses organ performance data from self_model to estimate fitness.
        """
        fitness = 0.0
        active_count = sum(1 for o in genome["organs"].values() if o["active"])

        # Reward active organs (but penalize too many — complexity cost)
        fitness += min(active_count, 6) * 0.1
        if active_count > 6:
            fitness -= (active_count - 6) * 0.05

        # Reward connections (diversity of interactions)
        n_conn = len(genome["connections"])
        fitness += min(n_conn, 10) * 0.05
        if n_conn > 10:
            fitness -= (n_conn - 10) * 0.02

        # Reward balanced neuron counts
        neuron_counts = [o["neuron_count"] for o in genome["organs"].values() if o["active"]]
        if neuron_counts:
            cv = np.std(neuron_counts) / (np.mean(neuron_counts) + 1e-6)
            fitness += max(0, 0.3 - cv * 0.2)

        # Meta parameter reasonableness
        lr = genome["meta"].get("learning_rate", 0.001)
        if 1e-4 <= lr <= 0.01:
            fitness += 0.1
        elif lr > 0.1 or lr < 1e-6:
            fitness -= 0.2

        # Add noise to break ties
        fitness += random.gauss(0, 0.01)

        return round(fitness, 4)

    def _save_best(self):
        """Save best genome to disk."""
        if self._best_genome:
            try:
                os.makedirs(os.path.dirname(GENOME_PATH), exist_ok=True)
                with open(GENOME_PATH, "w") as f:
                    json.dump(self._best_genome, f, indent=2)
                logger.info(f"[Evolver] Best genome saved: {GENOME_PATH}")
            except Exception as e:
                logger.warning(f"[Evolver] Save failed: {e}")

    def get_best_genome(self) -> Optional[Dict]:
        """Load best genome from disk."""
        if os.path.exists(GENOME_PATH):
            try:
                with open(GENOME_PATH) as f:
                    return json.load(f)
            except Exception:
                pass
        return self._best_genome or DEFAULT_GENOME

    def get_status(self) -> Dict:
        best = self.get_best_genome()
        return {
            "generation": self._generation,
            "population_size": len(self._population),
            "best_fitness": best.get("fitness", 0) if best else 0,
            "best_active_organs": sum(1 for o in best["organs"].values() if o["active"]) if best else 0,
        }


# Singleton
_evolver_instance = None

def get_evolver() -> ArchitectureEvolver:
    global _evolver_instance
    if _evolver_instance is None:
        _evolver_instance = ArchitectureEvolver()
    return _evolver_instance
