# SortingEnv (Manual) 📕

## Concept 📝
- Constant input with random or seasonal composition
- Discrete input batch runs over conveyor belt and sorting machine into storage
- Sorting machine has individual accuracy per material, depending on belt speed and load

![alt text](../docs/Sorting_Flowchart.svg)

- Reference: A Study showed that the sorting performance of SBS units is significantly influenced by both **throughput** (here "Quantity Sensor") and **input composition** (here "Material Sensor") (Kroell et al. 2023)

--- 
## Components ⚙️
**Input:**
- Discrete composition of materials A and B
  - Various material compositions (A/B)
  - Various material quantities
    - Random walk, seasonal variations, supply bottlenecks, ...

**Quantity Sensor**
- Provides rough information about the quantity of materials on the belt (e.g., 90% accuracy)

**Conveyor Belt**
- Speed adjustable by the agent
  - Influences the accuracy of the sorting machine
  - Agent receives information from the "Quantity Sensor" about the rough input quantity
  - 10% minimum, 100% maximum
    - Can be slower (e.g., with a lot of material) or faster (e.g., with less material)

**Material Sensor (Optional)**
- Provides rough information about the material composition on the belt
- Usable by the agent to adjust the sorting machine (positive/negative sorting)

**Sorting Machine**
- Sorts materials A and B
- Accuracy of the following sorting ("batch") depends on:
  - Current speed of the incoming batch on the input belt ("belt_speed")
  - Current quantity of the material of the batch on the belt ("belt_occupancy")
  - Type of selection: positive (rougher) or negative (cleaner) (cf. Kroell et al. 2023)
  - ❓ Material composition (e.g., "accuracy for B decreases if more A is present")

**Storage:**
- Stores materials A and B
- Discrete unlimited capacity (e.g. connection point for ContainerGym, see Pendyala et al. 2024)
- Accuracy sensor provides feedback on the current purity of the sorts

---
## RL Environment 🤖
**Action Space:**
- Speed of the belt (basic)
  - Influences general accuracy of the sorting machine (range of the sorting machine)
- Adjustment of sensor specificity (optional)
  - Temporarily higher accuracy for one material, lower for the other
  - Useful if the sensor has previously determined that this material will appear more frequently
  - Cleaner (negative), rougher (positive)

**Observation Space:** 
- Purity of sorts: A, B (basic)
  - (mis-sorted materials / correctly sorted materials)
- Sensor over the belt (optional)
  - Provides rough estimate of material composition
  - "Product purity decreases linearly with increasing material quantity." (Kroell et al. 2023)
  - "Higher target shares lead to higher purity." (Kroell et al. 2023)

**Reward:**   
- Purity and yield as targets (cf. Kroell et al. 2023)
  - ".. there is always a trade-off between high purity and high yield.." (Kroell et al. 2023)

1. **Yield**: Speed of sorting, faster = better (bonus for higher belt speed)
2. **Purity**: Optimal purity of sorts: average of current sorting accuracy

- ❓ Low power consumption (e.g., more at higher belt speeds)

---

## Futures Ideas 🔮
- [ ] Connect multiple sorting modules in series
- [ ] Multiple sensors -> Simulate failures of individual sensors
- [ ] Intentional disturbance of the belt (e.g., material falls off)
  - [ ] Agent should generate a warning
- [ ] Simulated customer orders: speed vs. accuracy
- [ ] Two input belts each with a sensor, one occasionally fails
- [ ] Additional materials with different accuracies

---

## References
- Kroell, N., Maghmoumi, A., Dietl, T., Chen, X., Küppers, B., Scherling, T., Feil, A., Greiff, K.: Towards digital twins of waste sorting plants: Developing data-driven process models of industrial-scale sensor-based sorting units by combining machine learning with near-infrared-based process monitoring. Resour. Conserv. Recycl. 200, 107257 (2024).
- Pendyala, A., Dettmer, J., Glasmachers, T., Atamna, A.: ContainerGym: A Real-World Reinforcement Learning Benchmark for Resource Allocation. In: Machine Learning, Optimization, and Data Science. pp. 78–92. Springer Nature Switzerland, Cham (2024).

---