# Releasing PsyStat

This guide is for the maintainer. A release publishes a Windows installer (+ portable zip) and two macOS disk images
(Apple Silicon and Intel) on the GitHub **Releases** page.

## One-time setup

1. Make sure the repository contains the files from this release package (`psystat.py`, `resources.py`, `psystat.spec`,
   `installer/`, `docs/`, `examples/`, `assets/icon.png` ...).
2. Put your application icon at `assets/icon.png` (square, 1024 x 1024). Run `python tools/make_icons.py` to create
   `assets/icon.ico` (Windows) and `assets/icon.icns` (macOS), and commit all three. Builds work without them but then
   use the default icon.
3. In **Settings → Actions → General → Workflow permissions**, choose **Read and write permissions**
   (needed to create releases).
4. Optional: enable **Settings → Security → Private vulnerability reporting**, and **Discussions** for questions.

## For each release

1. **Version.** Edit `APP_VERSION` in `psystat.py` (the single source of the version number) and add a section for the
   new version at the top of `CHANGELOG.md` (the release notes are taken from it). Update `version` and `date-released`
   in `CITATION.cff` and the version in the citation text in `README.md` and `docs/USER_MANUAL.md`.
2. **Test locally.**
   ```bash
   pip install -r requirements-dev.txt
   pytest
   python psystat.py --selftest
   ```
3. **Rebuild the documentation** if the manual changed: `python tools/build_docs.py` (needs `wkhtmltopdf` for the PDF).
   Commit the regenerated `docs/USER_MANUAL.html` and `.pdf`.
4. **Commit and push to `main`.** The *CI* workflow must be green on Windows, macOS and Linux.
5. **Tag and push:**
   ```bash
   git tag v1.2.0
   git push origin v1.2.0
   ```
   The tag must equal `v` + `APP_VERSION`, otherwise the workflow stops.
6. Open **Actions → Build release**. It builds the four downloads, runs `PsyStat --selftest` on each built application
   (the build fails if a library is missing from the bundle), and publishes the release with `SHA256SUMS.txt`.
7. **Download and test** the installer and a disk image on a clean Windows PC and on a Mac before announcing the release.

To build without publishing, run the workflow manually (**Actions → Build release → Run workflow**): the files are
attached to the run as artifacts.

## If a build fails

| Symptom | Likely cause and remedy |
|---|---|
| `--selftest` reports `FAIL load <package>` | The package is imported lazily and PyInstaller did not see it. Add it to `HIDDEN` (and, if it has data files, to the `collect_data_files` list) in `psystat.spec`. |
| Self-test: `bundled documents ... not found` | A file is missing from `DATAS` in `psystat.spec` or was not committed (check the PDF/HTML were rebuilt). |
| The macOS Intel job never starts | GitHub renamed or retired the Intel runner. Update the `runner:` label in `build-release.yml` (see GitHub's list of hosted runners). |
| Inno Setup not found | The `choco install innosetup` step failed; re-run the job. |
| App opens then closes on macOS (Apple Silicon) | The build must be signed; `installer/build_macos_dmg.sh` applies an ad-hoc signature. |

## Windows and macOS warnings (code signing)

The release is **not** signed with a paid certificate, so users see a SmartScreen (Windows) or Gatekeeper (macOS) prompt on
first launch; the manual explains how to continue. To remove these prompts you would need:

* **Windows:** an Authenticode code-signing certificate (or Azure Trusted Signing) and a `signtool` step before building the installer.
* **macOS:** an Apple Developer Program membership (paid), a *Developer ID Application* certificate, and a `codesign` +
  `xcrun notarytool submit --wait` + `xcrun stapler staple` step. Store the credentials as repository **secrets**.

## Dependencies and the lock file

`requirements-lock.txt` pins the exact versions used for the builds. To update them: create a clean virtual environment,
install `requirements.txt` (newest versions), run `pytest` and `python psystat.py --selftest`, then copy the versions from
`pip freeze` into `requirements-lock.txt`. Do this before a release rather than during one.

## Licensing reminder

The binaries include GPL-licensed components (for example PyQt6). See `NOTICE.md` before changing how the application is distributed.
