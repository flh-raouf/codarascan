# Barcode Pipeline TUI — Product Requirements Document

## Problem Statement

Running the repository's barcode pipelines currently requires remembering or copying long, pipeline-specific commands. Each pipeline supports a different combination of arguments, defaults, output behavior, and optional artifacts. This creates avoidable friction during development: commands are easy to mistype, supported options are difficult to discover, and each execution must be assembled manually.

The repository needs a focused terminal interface that makes a single pipeline run easy to configure, inspect, launch, monitor, stop, and trace without changing the pipelines into a new execution platform. The interface is intended primarily for the developer working on these pipelines. Result comparison and scientific evaluation remain deliberate human activities outside the interface.

## Solution

Build a repository-specific terminal user interface with OpenTUI and Bun. The application will be launched from the repository root with `bun run tui` and will guide the developer through selecting one PDF or image from a fixed input directory, selecting one supported pipeline, configuring only the options that apply to it, reviewing the exact generated command, and starting one foreground run.

The TUI will provide a concise live summary and complete raw logs. The Python pipelines will optionally emit a shared JSON Lines progress protocol so the interface can display reliable page progress, detected result counts, generated artifacts, and failures without scraping human-readable logs.

Every run will receive a unique directory beneath `data/output/`, using the structure `<date>/<document>/<pipeline>/<timestamp>/`. A `run-manifest.json` file will preserve the configuration and outcome for traceability. The TUI will not provide history browsing, reruns, comparisons, queues, or background execution.

The hybrid pipeline and all Dynamsoft/DBR functionality are excluded. The six supported pipelines are the deterministic locator, ZXing-only decoder, format-aware 2D-first ZXing pipeline, optimized decoder, localization-only pipeline, and CPU coarse-to-fine localization pipeline.

## User Stories

1. As a pipeline developer, I want to launch the TUI with `bun run tui`, so that I have one memorable entry point for pipeline execution.
2. As a pipeline developer, I want the application to verify that it was launched from the repository root, so that relative pipeline and data paths remain predictable.
3. As a pipeline developer, I want a clear preflight error when Bun, the Python virtual environment, an input directory, or a pipeline entry point is unavailable, so that I can fix the environment without diagnosing a confusing child-process failure.
4. As a pipeline developer, I want setup failures to show suggested commands without automatically changing my environment, so that I retain control over dependencies.
5. As a pipeline developer, I want the input picker restricted to `data/pdf/`, so that document selection is fast and consistent.
6. As a pipeline developer, I want the input picker to find supported files recursively, so that I can organize source documents into subdirectories.
7. As a pipeline developer, I want to choose one PDF or image per run, so that each execution has a simple and unambiguous input.
8. As a pipeline developer, I want unsupported files hidden from the picker, so that I cannot accidentally launch a pipeline with an obviously invalid input type.
9. As a pipeline developer, I want paths outside the fixed input directory rejected, so that the picker cannot escape its intended scope.
10. As a pipeline developer, I want to select one of the six supported pipelines, so that I can use the repository's relevant open-source execution paths from one interface.
11. As a pipeline developer, I want the hybrid Dynamsoft/DBR pipeline omitted, so that the interface does not expose an unsupported licensed engine.
12. As a pipeline developer, I want each pipeline described briefly in the selector, so that I can distinguish localization, decoding, and optimized variants.
13. As a pipeline developer, I want only options supported by the selected pipeline to be visible, so that I do not configure meaningless combinations.
14. As a pipeline developer, I want common options separated from advanced tuning options, so that routine execution remains quick while detailed controls stay available.
15. As a pipeline developer, I want the TUI to apply the validated defaults documented in the repository README, so that normal runs match the known command recipes.
16. As a pipeline developer, I want formats or localization kinds offered only where supported, so that the generated command is valid for the selected pipeline.
17. As a pipeline developer, I want page selection entered as a raw expression such as `8,16,19,21`, so that targeted runs remain compact.
18. As a pipeline developer, I want worker count exposed only on pipelines that support it, so that concurrency can be tuned without inventing unsupported flags.
19. As a pipeline developer, I want positive labels such as “Save crops” and “Save overlays,” so that I do not have to reason about inverted CLI flags.
20. As a pipeline developer, I want advanced settings for DPI, thresholds, minimum lengths, detector profiles, and recovery modes where applicable, so that specialized experiments remain possible.
21. As a pipeline developer, I want an advanced-arguments field, so that newly added or uncommon Python flags can be used before the TUI registry is updated.
22. As a pipeline developer, I want advanced arguments tokenized without invoking a shell, so that spaces can be handled without introducing shell injection.
23. As a pipeline developer, I want input and output overrides rejected in advanced arguments, so that the TUI remains responsible for run isolation.
24. As a pipeline developer, I want duplicates of form-controlled options rejected, so that the final configuration has one source of truth.
25. As a pipeline developer, I want unknown advanced flags passed to Python for final validation, so that the TUI does not unnecessarily block recent pipeline capabilities.
26. As a pipeline developer, I want the exact generated command and output directory shown before execution, so that I can verify what will run.
27. As a pipeline developer, I want a confirmation step before starting, so that an expensive pipeline is not launched accidentally.
28. As a pipeline developer, I want a dry-run action, so that I can inspect or copy the command without launching it.
29. As a pipeline developer, I want each execution written to a unique timestamped directory, so that previous artifacts are never silently replaced.
30. As a pipeline developer, I want folder names based on the date, document, pipeline, and local timestamp, so that output is understandable from the filesystem alone.
31. As a pipeline developer, I want exactly one run active at a time, so that the interface and resource usage remain straightforward.
32. As a pipeline developer, I want a Summary view during execution, so that I can see status, elapsed time, current page, total progress, result counts, artifacts, and errors at a glance.
33. As a pipeline developer, I want a Logs view containing complete stdout and stderr, so that no diagnostic information is hidden.
34. As a pipeline developer, I want errors surfaced in the Summary view even when I am not watching Logs, so that failures are immediately visible.
35. As a pipeline developer, I want structured progress produced independently of ordinary log wording, so that TUI summaries remain reliable as human-readable messages change.
36. As a command-line user, I want structured progress to be opt-in, so that existing pipeline output remains unchanged outside the TUI.
37. As a pipeline developer, I want the interface to tolerate malformed or ordinary output mixed with structured events, so that one bad line does not crash the run view.
38. As a pipeline developer, I want to stop the active run, so that I can end an incorrect or excessively slow execution.
39. As a pipeline developer, I want stopping to terminate the complete process group, so that worker processes are not orphaned.
40. As a pipeline developer, I want graceful termination attempted before forceful termination, so that pipelines can clean up when possible.
41. As a pipeline developer, I want a visible five-second termination countdown followed by automatic forceful termination, so that stopping has predictable behavior.
42. As a pipeline developer, I want closing the TUI to make a best-effort attempt to terminate the active process group, so that foreground work normally does not outlive the interface.
43. As a pipeline developer, I want partial artifacts retained after failure or cancellation, so that they remain available for diagnosis.
44. As a pipeline developer, I want each run marked as succeeded, failed, or cancelled, so that its outcome is unambiguous.
45. As a pipeline developer, I want every run to contain a manifest, so that its execution can be understood without relying on terminal scrollback.
46. As a pipeline developer, I want the manifest to record the pipeline, repository-relative input, output location, exact executable, argument array, readable command, timestamps, duration, exit status, processed pages, result counts, and artifact paths, so that the run is traceable.
47. As a pipeline developer, I want UTC timestamps inside the manifest, so that machine-readable times are unambiguous.
48. As a pipeline developer, I want Algiers local time used in the interface and output folder name, so that filesystem navigation matches my local working context.
49. As a pipeline developer, I want manifests to exclude Git state, dependency inventories, and resource statistics, so that they remain focused and compact.
50. As a keyboard user, I want every workflow to be fully usable without a mouse, so that the TUI works consistently in terminal environments.
51. As a pipeline developer, I want mouse support only when OpenTUI provides it with negligible implementation complexity, so that optional convenience does not expand the project scope.
52. As a maintainer, I want pipeline definitions registered explicitly in one typed catalog, so that supported behavior is visible and intentional.
53. As a maintainer, I want adding or changing a pipeline definition isolated from presentation code, so that CLI evolution is easy to maintain.
54. As a maintainer, I want process supervision isolated behind a small interface, so that launching and cancellation can be tested independently of terminal rendering.
55. As a maintainer, I want output and manifest lifecycle management isolated behind a small interface, so that filesystem guarantees can be tested independently.
56. As a maintainer, I want the progress event contract shared across supported Python pipelines, so that the TUI does not maintain six incompatible parsers.

## Implementation Decisions

- The application will use OpenTUI with TypeScript and Bun.
- The only supported launch command for version 1 is `bun run tui` from the repository root.
- The application is repository-specific and will not attempt to become a generic pipeline runner.
- Six pipelines will be explicitly registered. The hybrid pipeline and Dynamsoft/DBR engine will not be registered or represented.
- The pipeline registry will be a deep module that owns labels, descriptions, entry points, README-derived defaults, capabilities, option definitions, argument mappings, and reserved flags.
- The command builder will accept a validated pipeline selection and form configuration and return an executable plus an argument array. It will not return a shell command for execution.
- A human-readable command will be derived for preview and manifest purposes only.
- Advanced arguments will use shell-style tokenization without shell evaluation.
- Input, output, and form-controlled duplicate options will be rejected in advanced arguments. Other unknown options will be forwarded for Python to validate.
- The document catalog will recursively enumerate supported PDF and image formats beneath the fixed input root and enforce directory containment.
- One document and one pipeline will be selected per run. No queue or parallel execution model will be introduced.
- The TUI will use README defaults rather than inventing unified defaults. Any explicit arguments added by the TUI will remain visible in the command preview.
- Unsupported fields will disappear when the selected pipeline changes.
- Common controls will be shown directly. Detector internals and uncommon tuning controls will live in an expandable advanced-settings section.
- The launch flow will include command preview, output preview, confirmation, and dry run.
- Run directories will be unique and follow `data/output/<date>/<document>/<pipeline>/<timestamp>/`.
- Folder timestamps will use the Africa/Algiers timezone. Manifest timestamps will use UTC.
- The run workspace manager will own safe slug creation, unique directory allocation, and manifest state transitions.
- Each run will create one `run-manifest.json`. Manifests are filesystem artifacts and will not be indexed by the TUI.
- The manifest will record configuration, timing, status, exit information, processed pages, counts, and artifacts. It will exclude Git metadata, Python dependency versions, CPU metrics, memory metrics, and resource time series.
- The process supervisor will launch the pipeline in a dedicated process group and expose output events, completion, and cancellation through a small interface.
- Cancellation will send `SIGTERM` to the process group, display a five-second countdown, and automatically send `SIGKILL` if the group remains alive.
- TUI exit and catchable crashes will trigger best-effort process-group termination. The product will not claim termination guarantees after uncatchable termination such as `kill -9`.
- Partial output will never be automatically deleted after failure or cancellation.
- Supported Python pipelines will accept an optional structured-progress mode and emit newline-delimited JSON events while preserving their ordinary output behavior when the mode is absent.
- The progress protocol will cover run lifecycle, page lifecycle, detected results, written artifacts, warnings, and failures. Events will be versioned or designed for additive evolution.
- Structured progress will be written to a clearly defined stream and parsed independently of ordinary logs. Malformed event lines will be preserved as diagnostic output and will not terminate the TUI.
- The run screen will provide Summary and Logs tabs. The Summary tab will show status, elapsed time, page progress, result counts, artifact activity, and surfaced errors. Resource usage is excluded.
- Preflight checks will diagnose missing runtime prerequisites and show remediation guidance without installing dependencies.
- Keyboard navigation is required for all functions. Mouse-specific behavior will not receive dedicated engineering effort.

## Testing Decisions

- Tests will assert observable contracts and outcomes rather than internal implementation details or terminal styling.
- The pipeline registry and command builder should be tested as a deep unit: each of the six registrations must produce the intended README-default command and correctly map supported form values.
- Validation tests should cover reserved arguments, duplicates, unknown forwarded flags, quoting, whitespace, paths containing spaces, and arguments that would be dangerous if interpreted by a shell.
- Document catalog tests should cover recursive discovery, supported and unsupported extensions, stable ordering, missing directories, symlinks, and attempts to escape the fixed input root.
- Run workspace tests should cover slug creation, local-date layout, timestamp collisions, guaranteed unique allocation, and refusal to reuse an existing directory silently.
- Manifest tests should exercise externally visible state transitions for pending, running, succeeded, failed, and cancelled runs, including persistence of partial information after interruption.
- Progress parser tests should cover valid events, unknown additive fields, unknown event types, malformed JSON, ordinary stdout/stderr, partial chunks, and multiple events arriving in one chunk.
- Process supervisor tests should use controlled fixture processes to verify stdout/stderr streaming, exit-code propagation, graceful cancellation, five-second escalation behavior through an injectable clock, and complete process-group termination.
- Python progress-contract tests should verify that each included pipeline accepts the opt-in mode and emits protocol-conforming lifecycle events without changing default CLI output.
- A small integration test should execute a harmless fixture pipeline through the complete command, supervisor, event, output, and manifest flow.
- TUI tests should be limited to major observable state transitions: startup preflight, document selection, pipeline selection, configuration, confirmation, dry run, active run, cancellation, success, and failure.
- Pixel-perfect or large terminal snapshot tests are not required because they are brittle and provide little confidence in execution correctness.
- The repository currently has no established automated unit-test suite for this layer. Existing regression utilities provide prior art for outcome-oriented pipeline validation, but the new TypeScript orchestration modules will require a new Bun-compatible test setup and the Python progress contract will require focused Python tests.

## Out of Scope

- The hybrid barcode pipeline and Dynamsoft/DBR support.
- Batch input selection or directory-wide execution.
- Running more than one pipeline per execution.
- Parallel pipelines, job queues, and scheduled runs.
- Presets and saved configurations.
- Run-history indexing, browsing, searching, or filtering.
- Rerunning a historical manifest from the TUI.
- Automated comparison, scoring, ranking, benchmarking, or regression analysis.
- CPU, memory, or other resource monitoring.
- Resource time-series collection.
- PDF thumbnails and previews.
- Viewing generated crops or overlays inside the terminal.
- Background, detached, or recoverable jobs.
- Remote execution.
- Automatic dependency installation or environment mutation.
- Automatic pipeline discovery or CLI introspection.
- A plugin system or general-purpose runner architecture.
- SQLite or any other history database.
- Dedicated mouse interaction design.
- Launching from arbitrary working directories.
- Editing pipeline source code from the TUI.

## Further Notes

- The fixed input root is `data/pdf/`, despite accepting both PDFs and supported images. The name is intentional for version 1.
- The fixed output root is `data/output/`.
- Human comparison of pipeline outputs remains outside the product. The manifest exists for traceability, not for building a comparison system.
- Several pipelines currently expose different flags and defaults. The explicit registry is the compatibility boundary that keeps these differences out of presentation code.
- Reliable current-page progress and artifact counts require cooperation from the Python processes. The structured protocol is therefore part of version 1 rather than optional polish.
- The worktree contained unrelated existing modifications when this PRD was created. Implementation must preserve them and avoid treating them as part of this feature.
