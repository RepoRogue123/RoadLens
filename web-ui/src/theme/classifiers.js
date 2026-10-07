/**
 * Classifier roster — single source shared by the ledger and dashboard.
 * `measured` = trained on RealSense-measured depth; `ml` = trained on
 * Depth-Anything pseudo-labels; `rule` = hand-set thresholds.
 */
export const CLASSIFIERS = [
  { id: 'metric_depth', name: 'Depth Model (measured)', type: 'measured' },
  { id: 'rule_based', name: 'Rule-Based', type: 'rule' },
  { id: 'logistic_regression', name: 'Logistic Regression', type: 'ml' },
  { id: 'random_forest', name: 'Random Forest', type: 'ml' },
  { id: 'svm', name: 'SVM (RBF Kernel)', type: 'ml' },
  { id: 'naive_bayes', name: 'Naive Bayes', type: 'ml' },
];

/** API classification name → roster id. */
export const NAME_TO_ID = {
  'Depth Model (measured)': 'metric_depth',
  'Rule-Based': 'rule_based',
  'Logistic Regression': 'logistic_regression',
  'Random Forest': 'random_forest',
  'SVM (RBF Kernel)': 'svm',
  'Naive Bayes': 'naive_bayes',
};
