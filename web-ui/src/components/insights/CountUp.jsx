import { useEffect, useRef } from 'react';
import { animate, useInView } from 'motion/react';

/**
 * A number that counts up from zero the first time it scrolls into view.
 * Writes straight to the DOM node so the animation does not re-render React.
 */
export default function CountUp({ value, decimals = 0, duration = 1.4, prefix = '', suffix = '' }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: '-40px' });
  const format = (v) => `${prefix}${v.toFixed(decimals)}${suffix}`;

  useEffect(() => {
    if (!inView || value == null || !ref.current) return undefined;
    const controls = animate(0, value, {
      duration,
      ease: [0.22, 1, 0.36, 1],
      onUpdate: (v) => {
        if (ref.current) ref.current.textContent = format(v);
      },
    });
    return () => controls.stop();
    // format depends only on prefix/suffix/decimals, listed below
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inView, value, duration, prefix, suffix, decimals]);

  if (value == null) return <span>—</span>;
  return <span ref={ref}>{format(0)}</span>;
}
