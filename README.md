# EnergyTradingRL
This repository provides a reproducible toolkit for grid-aware EV charging with DRL. 

It includes:
(i) a flexibility pre-training module (LoadManagerMDP + CentralEnergyManager) 
(ii) a station-level simulator (ChargingStationEnv) with PV, grid limits, community batteries, and EV arrivals/departures. 
We implement PPO/TD3/DDPG under a controlled protocol (identical inputs/normalization, fixed 48-step episodes, matched seeds,
deterministic evaluation with frozen stats) and export ready-to-use artifacts (CSV logs, MAE/cost/reward/flexibility plots, 
multi-scenario summaries). The goal is to make fair comparisons and replication easy—clone, set the seeds, run the scenarios, 
and reproduce the figures/tables reported in the paper.

# Reference Dataset: 
F. Rodrigues, C. Cardeira, J. M. F. Calado, and R. Melício, “Load Profile Analysis Tool for Electrical Appliances in Households Assisted by CPS,” Energy Procedia, vol. 106, pp. 215–224, Dec. 2016, doi: https://doi.org/10.1016/j.egypro.2016.12.117.

# Reference Paper:
M. Asim. Amin, R. Procopio, M. Invernizzi, A. Bonfiglio, and Y. Jia, “Exploring the role of energy Communities: A comprehensive review,” Energy Conversion and Management X, pp. 100883–100883, Jan. 2025, doi: https://doi.org/10.1016/j.ecmx.2025.100883.

‌

‌
