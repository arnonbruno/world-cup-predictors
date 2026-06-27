# Hypothesized Causal DAG

```mermaid
graph LR
    GDP[GDP / Income] --> Elo[Elo / Team Quality]
    Pop[Population / Talent Pool] --> Elo
    Tradition[Football Tradition] --> Elo
    Titles[Past WC Titles] --> Tradition
    Host[Hosting] --> Win[World Cup Win]
    Elo --> Win
    Institution[Institutions / Governance] --> Econ[Macro Conditions]
    Econ --> GDP
    Region[Geography / Confederation] --> Elo
    Region --> Win
```
