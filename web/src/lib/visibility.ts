/// The rule both provers follow about hidden tabs, in one place so they cannot drift apart.
///
/// Chrome batches timers to once a second in a hidden tab and once a minute after five
/// minutes of it. That does not slow the arithmetic, but it moves the clock the arithmetic is
/// measured against, so a rep that spans a hidden tab is not a measurement. The rule is
/// therefore: wait for the tab before starting a rep, and throw the rep away if the tab went
/// away during it.
///
/// Both halves matter, and each side used to have only one of them. The snarkjs loop checked
/// before and after but `continue`d rather than retried, so a backgrounded run came back with
/// zero timed reps, and `median([])` is `NaN`: the verdict panel rendered "NaNx" with nothing
/// to say why. The GPU loop waited before a rep and then recorded whatever came out, hidden
/// tab or not, which is the same rule broken in the other direction.

/// Resolves immediately if the tab is visible, otherwise when it next becomes visible.
export function visible(): Promise<void> {
  if (document.visibilityState === 'visible') return Promise.resolve();
  return new Promise((res) => {
    const on = () => {
      if (document.visibilityState === 'visible') {
        document.removeEventListener('visibilitychange', on);
        res();
      }
    };
    document.addEventListener('visibilitychange', on);
  });
}

/// Counts discarded reps and refuses to retry forever.
///
/// `visible()` blocks rather than spins, so a tab left in the background cannot burn CPU
/// here. What this guards is the other case: a tab flipping in and out fast enough that every
/// rep spans a hide. Without a cap that re-proves indefinitely and the run never ends, which
/// from the outside is indistinguishable from a hang.
export class Discards {
  private n = 0;
  constructor(private readonly limit: number) {}

  /// Records one discarded rep. Throws once too many have piled up, naming the cause, since
  /// "the tab kept going to the background" is not something a stack trace would suggest.
  count(what: string): void {
    if (++this.n > this.limit) {
      throw new Error(
        `${what}: gave up after ${this.n} reps were discarded for spanning a hidden tab. ` +
          `Keep this tab in the foreground for the duration of the run.`
      );
    }
  }
}
