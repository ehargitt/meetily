import { expect, test } from 'bun:test';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

// The real RecordingStateContext and TranscriptContext can't be loaded in this
// process: other suites mock them for the whole run and bun cannot un-mock a
// module. The suite runs in a fresh bun process instead.
const frontendDir = join(dirname(fileURLToPath(import.meta.url)), '..', '..');

test('real recording contexts (isolated process)', () => {
  const result = Bun.spawnSync(
    [process.execPath, 'test', './tests/contexts/recording-contexts.isolated.tsx'],
    { cwd: frontendDir, stdout: 'pipe', stderr: 'pipe' },
  );
  const output = `${result.stdout.toString()}${result.stderr.toString()}`;
  if (result.exitCode !== 0) {
    throw new Error(`isolated recording-context suite failed:\n${output}`);
  }
  expect(output).toMatch(/\b4 pass\b/);
  expect(output).toMatch(/\b0 fail\b/);
}, 30000);
