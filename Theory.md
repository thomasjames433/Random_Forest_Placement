# Random Forest / Decision Tree

### Entropy
\[
E = - \sum p(w) \log_2 p(w)
\]

Probability:
\[
p(A) = \frac{\#a}{n}
\]

### Information Gain
\[
IG = E(\text{parent}) - \text{weighted avg}(E(\text{children}) )
\]

---

## Machine A
| Symbol | Probability |
|--------|-------------|
| A | 0.25 |
| B | 0.25 |
| C | 0.25 |
| D | 0.25 |

---

## Machine B
| Symbol | Probability |
|--------|-------------|
| A | 0.50 |
| B | 0.25 |
| C | 0.125 |
| D | 0.125 |

---

## Question
If we had to predict the next symbol from each machine, how many questions would you have to ask?

---

## Machine 1 Decision Tree
```
Is it AB?
  Yes → (AB)
        Is it A?
            Yes → A
            No  → B
  No  → (CD)
        Is it C?
            Yes → C
            No  → D
```
Uncertainty = 2 questions per symbol.

---

## Machine 2 Decision Tree
```
Is it A?
  Yes → A
  No → (B, C, D)
        Is it D?
            Yes → D
            No → Is it B?
                    Yes → B
                    No  → C
```

Uncertainty:
```
= p(A)*1 + p(B)*3 + p(C)*2 + p(D)*2
= 0.5×1 + 0.25×3 + 0.125×2 + 0.125×2
= 1.75
```
```
If we had 100 predictions. Machine takes 2*100 questions while Machine B would take 1.75 *100 questions
```

---

## Number of Questions (Information Theory)
For N items:
```
log₂(# outcomes) = log₂(1/p)
```

Thus:
\[
H = \sum p_i \times (\text{# of questions})
\]
which becomes
\[
H = \sum p \log_2 (1/p) = -\sum p \log_2 p.
\]

