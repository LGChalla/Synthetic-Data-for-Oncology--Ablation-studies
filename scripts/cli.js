#!/usr/bin/env node
/**
 * cli.js — StageCraft TCBB Ablation Runner
 *
 * Usage:
 *   node scripts/cli.js run rag           # Ablation 1: RAG vs No-RAG
 *   node scripts/cli.js run gate          # Ablation 2: Gate vs No-Gate
 *   node scripts/cli.js run decomp        # Ablation 3: Gate Decomposition
 *   node scripts/cli.js run all           # All three in sequence
 *   node scripts/cli.js status            # Show what has run / what's pending
 *   node scripts/cli.js results           # Print result CSVs as tables
 *   node scripts/cli.js push [message]    # Commit + push current state to GitHub
 */

import { program }        from 'commander';
import chalk              from 'chalk';
import ora                from 'ora';
import Table              from 'cli-table3';
import { spawn }          from 'child_process';
import { execSync }       from 'child_process';
import fs                 from 'fs';
import path               from 'path';
import { fileURLToPath }  from 'url';
import { createReadStream, existsSync, mkdirSync, readdirSync, statSync } from 'fs';
import readline           from 'readline';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT      = path.resolve(__dirname, '..');

// ── Paths ─────────────────────────────────────────────────────────────────────
const ABLATIONS_DIR = path.join(ROOT, 'ablations');
const RESULTS_DIR   = path.join(ROOT, 'results');
const LOGS_DIR      = path.join(ROOT, 'logs');

const ABLATION_MAP = {
  rag:    {
    label:  'Ablation 1 — RAG vs. No-RAG',
    script: path.join(ABLATIONS_DIR, 'ablation_rag_vs_norag.py'),
    outDir: path.join(RESULTS_DIR, 'ablation_rag'),
    logFile: path.join(LOGS_DIR, 'ablation_rag.log'),
    evalCsv: path.join(RESULTS_DIR, 'ablation_rag', 'evaluation_results.csv'),
    expLog:  path.join(RESULTS_DIR, 'ablation_rag', 'experiment.log'),
  },
  gate:   {
    label:  'Ablation 2 — Gate vs. No-Gate',
    script: path.join(ABLATIONS_DIR, 'ablation_gate_vs_nogate.py'),
    outDir: path.join(RESULTS_DIR, 'ablation_gate'),
    logFile: path.join(LOGS_DIR, 'ablation_gate.log'),
    evalCsv: path.join(RESULTS_DIR, 'ablation_gate', 'evaluation_results.csv'),
    expLog:  path.join(RESULTS_DIR, 'ablation_gate', 'experiment.log'),
  },
  decomp: {
    label:  'Ablation 3 — Gate Decomposition',
    script: path.join(ABLATIONS_DIR, 'ablation_gate_decomposition.py'),
    outDir: path.join(RESULTS_DIR, 'ablation_gate_decomp'),
    logFile: path.join(LOGS_DIR, 'ablation_gate_decomp.log'),
    evalCsv: path.join(RESULTS_DIR, 'ablation_gate_decomp', 'evaluation_results.csv'),
    expLog:  path.join(RESULTS_DIR, 'ablation_gate_decomp', 'experiment.log'),
  },
};

// ── Ensure dirs exist ─────────────────────────────────────────────────────────
[LOGS_DIR, RESULTS_DIR].forEach(d => mkdirSync(d, { recursive: true }));

// ─────────────────────────────────────────────────────────────────────────────
// RUNNER — spawns Python, streams stdout/stderr, writes log file
// ─────────────────────────────────────────────────────────────────────────────

function runPythonScript(scriptPath, extraArgs = [], logFile) {
  return new Promise((resolve, reject) => {
    mkdirSync(path.dirname(logFile), { recursive: true });
    const logStream = fs.createWriteStream(logFile, { flags: 'a' });
    const startTime = new Date();

    logStream.write(`\n${'='.repeat(60)}\n`);
    logStream.write(`START: ${startTime.toISOString()}\n`);
    logStream.write(`SCRIPT: ${scriptPath}\n`);
    logStream.write(`ARGS: ${extraArgs.join(' ')}\n`);
    logStream.write(`${'='.repeat(60)}\n\n`);

    const proc = spawn('python3', [scriptPath, ...extraArgs], {
      cwd:   ROOT,
      env:   { ...process.env, PYTHONUNBUFFERED: '1' },
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    // Stream stdout — both to console and log file
    proc.stdout.on('data', (chunk) => {
      const text = chunk.toString();
      process.stdout.write(text);
      logStream.write(text);
    });

    // Stream stderr — colour red on console, raw to log
    proc.stderr.on('data', (chunk) => {
      const text = chunk.toString();
      process.stderr.write(chalk.red(text));
      logStream.write(`[STDERR] ${text}`);
    });

    proc.on('close', (code) => {
      const elapsed = ((Date.now() - startTime) / 1000 / 60).toFixed(1);
      const status  = code === 0 ? 'SUCCESS' : `FAILED (exit ${code})`;
      logStream.write(`\n${'='.repeat(60)}\n`);
      logStream.write(`${status} — elapsed ${elapsed} min\n`);
      logStream.end();

      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`${path.basename(scriptPath)} exited with code ${code}`));
      }
    });

    proc.on('error', (err) => {
      logStream.write(`\n[SPAWN ERROR] ${err.message}\n`);
      logStream.end();
      reject(err);
    });
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// STATUS — inspects output directories to show what has run
// ─────────────────────────────────────────────────────────────────────────────

function getStatus(key) {
  const cfg = ABLATION_MAP[key];
  const states = {
    evalDone:    existsSync(cfg.evalCsv),
    hasExpLog:   existsSync(cfg.expLog),
    hasOutDir:   existsSync(cfg.outDir),
    hasLogFile:  existsSync(cfg.logFile),
  };

  // Count adapters saved
  const adapterDirs = ['adapter_rag', 'adapter_norag',
                       'adapter_nogate', 'adapter_gate',
                       'adapter_S', 'adapter_SO', 'adapter_SOC'];
  const savedAdapters = adapterDirs.filter(d =>
    existsSync(path.join(cfg.outDir, d, 'final_adapter'))
  );

  // Last event from experiment log
  let lastEvent = '—';
  if (states.hasExpLog) {
    try {
      const lines = fs.readFileSync(cfg.expLog, 'utf8').trim().split('\n');
      const last  = JSON.parse(lines[lines.length - 1]);
      lastEvent   = `${last.event} @ ${last.timestamp?.slice(0, 19) ?? ''}`;
    } catch {}
  }

  return { ...states, savedAdapters, lastEvent };
}

function printStatusTable() {
  const table = new Table({
    head: [
      chalk.cyan('Ablation'),
      chalk.cyan('Out Dir'),
      chalk.cyan('Adapters'),
      chalk.cyan('Eval Done'),
      chalk.cyan('Last Event'),
    ],
    colWidths: [34, 10, 14, 12, 48],
  });

  for (const [key, cfg] of Object.entries(ABLATION_MAP)) {
    const s = getStatus(key);
    table.push([
      cfg.label,
      s.hasOutDir  ? chalk.green('✓') : chalk.gray('—'),
      s.savedAdapters.length > 0
        ? chalk.green(s.savedAdapters.join(', '))
        : chalk.gray('none'),
      s.evalDone   ? chalk.green('✓') : chalk.yellow('pending'),
      chalk.gray(s.lastEvent),
    ]);
  }

  console.log('\n' + chalk.bold('Ablation Status'));
  console.log(table.toString());
  console.log();
}

// ─────────────────────────────────────────────────────────────────────────────
// RESULTS — reads evaluation CSVs and prints as tables
// ─────────────────────────────────────────────────────────────────────────────

function parseCSV(filePath) {
  const text  = fs.readFileSync(filePath, 'utf8').trim();
  const lines = text.split('\n');
  const headers = lines[0].split(',');
  return lines.slice(1).map(line => {
    const vals = line.split(',');
    return Object.fromEntries(headers.map((h, i) => [h.trim(), vals[i]?.trim() ?? '']));
  });
}

function printResultsTable(key) {
  const cfg = ABLATION_MAP[key];
  if (!existsSync(cfg.evalCsv)) {
    console.log(chalk.yellow(`  [${key}] No evaluation results yet (${cfg.evalCsv})`));
    return;
  }

  const rows    = parseCSV(cfg.evalCsv);
  const colKeys = Object.keys(rows[0] ?? {});

  // Determine column widths (min 8, max 18)
  const widths = colKeys.map(k =>
    Math.min(18, Math.max(8, k.length + 2,
      ...rows.map(r => String(r[k] ?? '').length + 2)))
  );

  const table = new Table({
    head:      colKeys.map(k => chalk.cyan(k)),
    colWidths: widths,
  });

  for (const row of rows) {
    table.push(colKeys.map(k => {
      const v = row[k] ?? '';
      // Colour accuracy values: green >0.5, yellow >0.3, red otherwise
      if (['accuracy','macro_f1'].includes(k)) {
        const n = parseFloat(v);
        if (!isNaN(n)) {
          if (n >= 0.5) return chalk.green(v);
          if (n >= 0.3) return chalk.yellow(v);
          return chalk.red(v);
        }
      }
      return v;
    }));
  }

  console.log(chalk.bold(`\n${cfg.label} — Evaluation Results`));
  console.log(table.toString());
}

// ─────────────────────────────────────────────────────────────────────────────
// GIT PUSH
// ─────────────────────────────────────────────────────────────────────────────

function gitPush(message) {
  const msg = message || `ablation results update ${new Date().toISOString().slice(0, 19)}`;
  console.log(chalk.cyan('\nCommitting and pushing to GitHub...'));
  try {
    execSync('git add -A', { cwd: ROOT, stdio: 'inherit' });
    execSync(`git commit -m "${msg}"`, { cwd: ROOT, stdio: 'inherit' });
    execSync('git push', { cwd: ROOT, stdio: 'inherit' });
    console.log(chalk.green('\n✓ Pushed successfully.'));
  } catch (err) {
    // git commit exits non-zero when there's nothing to commit
    if (err.message.includes('nothing to commit')) {
      console.log(chalk.yellow('Nothing new to commit.'));
    } else {
      console.error(chalk.red(`Git error: ${err.message}`));
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI DEFINITION
// ─────────────────────────────────────────────────────────────────────────────

program
  .name('ablate')
  .description('StageCraft TCBB Ablation Runner')
  .version('1.0.0');

// ── run ───────────────────────────────────────────────────────────────────────
program
  .command('run <ablation>')
  .description('Run one or all ablations. <ablation>: rag | gate | decomp | all')
  .option('--skip-gen',        'Skip generation; use existing corpus files')
  .option('--skip-train',      'Skip adapter training; run evaluation only')
  .option('--n-target <n>',    'Golden records per condition', '162')
  .option('--faiss-index <p>', 'Path to MedCPT FAISS index (ablation 1 only)')
  .option('--push',            'Auto-push to GitHub after completion')
  .action(async (ablation, opts) => {
    const keys = ablation === 'all'
      ? ['rag', 'gate', 'decomp']
      : [ablation];

    for (const key of keys) {
      if (!ABLATION_MAP[key]) {
        console.error(chalk.red(`Unknown ablation: ${key}. Use: rag | gate | decomp | all`));
        process.exit(1);
      }

      const cfg  = ABLATION_MAP[key];
      const args = [];

      if (opts.skipGen)      args.push('--skip-gen');
      if (opts.skipTrain)    args.push('--skip-train');
      if (opts.nTarget)      args.push('--n-target', opts.nTarget);
      if (opts.faissIndex && key === 'rag') args.push('--faiss-index', opts.faissIndex);

      console.log(chalk.bold(`\n${'═'.repeat(60)}`));
      console.log(chalk.bold(`  ${cfg.label}`));
      console.log(chalk.bold(`${'═'.repeat(60)}\n`));
      console.log(chalk.gray(`  Script  : ${cfg.script}`));
      console.log(chalk.gray(`  Log     : ${cfg.logFile}`));
      console.log(chalk.gray(`  Args    : ${args.join(' ') || '(none)'}\n`));

      const spinner = ora({
        text:    `Running ${cfg.label}...`,
        spinner: 'dots',
      }).start();

      try {
        await runPythonScript(cfg.script, args, cfg.logFile);
        spinner.succeed(chalk.green(`${cfg.label} — complete`));
      } catch (err) {
        spinner.fail(chalk.red(`${cfg.label} — FAILED: ${err.message}`));
        console.error(chalk.gray(`  See log: ${cfg.logFile}`));
        if (ablation !== 'all') process.exit(1);
        // For 'all', continue to next ablation even if one fails
      }
    }

    if (opts.push) {
      gitPush(`auto: ${ablation} ablation complete`);
    }
  });

// ── status ────────────────────────────────────────────────────────────────────
program
  .command('status')
  .description('Show which ablations have run and what outputs exist')
  .action(() => {
    printStatusTable();
  });

// ── results ───────────────────────────────────────────────────────────────────
program
  .command('results [ablation]')
  .description('Print evaluation results. <ablation>: rag | gate | decomp | all (default)')
  .action((ablation) => {
    const keys = (!ablation || ablation === 'all')
      ? ['rag', 'gate', 'decomp']
      : [ablation];
    for (const key of keys) {
      if (!ABLATION_MAP[key]) {
        console.error(chalk.red(`Unknown ablation: ${key}`));
        continue;
      }
      printResultsTable(key);
    }
  });

// ── push ──────────────────────────────────────────────────────────────────────
program
  .command('push [message]')
  .description('Commit all changes and push to GitHub')
  .action((message) => {
    gitPush(message);
  });

// ── logs ──────────────────────────────────────────────────────────────────────
program
  .command('logs <ablation>')
  .description('Tail the live log for an ablation. <ablation>: rag | gate | decomp')
  .option('-n <lines>', 'Number of recent lines to show', '40')
  .action((ablation, opts) => {
    const cfg = ABLATION_MAP[ablation];
    if (!cfg) {
      console.error(chalk.red(`Unknown ablation: ${ablation}`));
      process.exit(1);
    }
    if (!existsSync(cfg.logFile)) {
      console.log(chalk.yellow(`No log file yet: ${cfg.logFile}`));
      return;
    }
    // Print last N lines then watch
    const lines = fs.readFileSync(cfg.logFile, 'utf8').split('\n');
    const tail  = lines.slice(-parseInt(opts.n)).join('\n');
    console.log(chalk.gray(`── Last ${opts.n} lines of ${cfg.logFile} ──\n`));
    console.log(tail);
    console.log(chalk.gray('\n── Watching for new output (Ctrl-C to stop) ──'));

    // Watch for new content
    let pos = fs.statSync(cfg.logFile).size;
    setInterval(() => {
      const stat = fs.statSync(cfg.logFile);
      if (stat.size > pos) {
        const stream = fs.createReadStream(cfg.logFile, { start: pos });
        stream.on('data', chunk => process.stdout.write(chunk.toString()));
        pos = stat.size;
      }
    }, 1000);
  });

program.parse();
