# Third-party software and licensing notice

PsyStat's **own source code** is released under the [MIT License](LICENSE).

PsyStat is built on open-source libraries, each distributed under its own licence. The table lists the main ones.
**Licences can change between versions; please check each project's current licence before redistributing.**

| Component | Typical licence |
|---|---|
| PyQt6 (graphical interface) | GPL v3 *or* a commercial licence from Riverbank Computing |
| pingouin | GPL v3 |
| factor_analyzer | GPL v2 or later |
| NumPy, pandas, SciPy, statsmodels, scikit-learn, networkx, prince, semopy, openpyxl, mistune | BSD / MIT-style |
| matplotlib | matplotlib licence (BSD-compatible) |

## What this means in practice

* Because the **packaged applications** (the Windows installer and the macOS disk images) include GPL-licensed
  components such as PyQt6, they are distributed **under the terms of the GNU General Public License** as a combined work.
  The complete source code of PsyStat is public in this repository, which satisfies the source-availability requirement.
  The MIT licence of PsyStat's own code is compatible with this.
* If you **redistribute** modified PsyStat binaries, you must also make the corresponding source code available and keep
  the licence notices of all components.
* If you wish to build a **closed-source or commercially licensed** derivative, you must replace or relicense the GPL
  components (for example by using a commercial PyQt licence or switching to PySide6, which is LGPL) and replace
  GPL-licensed statistics libraries. Seek legal advice.

This notice is provided for information and is not legal advice.
