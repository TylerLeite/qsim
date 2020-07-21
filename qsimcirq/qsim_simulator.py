# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Sequence, Tuple

from cirq import (
  circuits,
  linalg,
  ops,
  protocols,
  sim,
  study,
  value,
  SimulatesAmplitudes,
  SimulatesFinalState,
  SimulatesSamples,
)

import numpy as np

from qsimcirq import qsim
import qsimcirq.qsim_circuit as qsimc


class QSimSimulatorState(sim.StateVectorSimulatorState):

    def __init__(self,
                 qsim_data: np.ndarray,
                 qubit_map: Dict[ops.Qid, int]):
      state_vector = qsim_data.view(np.complex64)
      super().__init__(state_vector=state_vector, qubit_map=qubit_map)


class QSimSimulatorTrialResult(sim.StateVectorTrialResult):

    def __init__(self,
                 params: study.ParamResolver,
                 measurements: Dict[str, np.ndarray],
                 final_simulator_state: QSimSimulatorState):
      super().__init__(params=params,
                       measurements=measurements,
                       final_simulator_state=final_simulator_state)


class QSimSimulator(SimulatesSamples, SimulatesAmplitudes, SimulatesFinalState):

  def __init__(self, qsim_options: dict = {}):
    if any(k in qsim_options for k in ('c', 'i')):
      raise ValueError(
          'Keys "c" & "i" are reserved for internal use and cannot be used in QSimCircuit instantiation.'
      )
    self.qsim_options = {'t': 1, 'v': 0}
    self.qsim_options.update(qsim_options)
    return

  def _run(
        self,
        circuit: circuits.Circuit,
        param_resolver: study.ParamResolver,
        repetitions: int
  ) -> Dict[str, np.ndarray]:
    """Run a simulation, mimicking quantum hardware.

    Args:
        program: The circuit to simulate.
        param_resolver: Parameters to run with the program.
        repetitions: Number of times to repeat the run.

    Returns:
        A dictionary from measurement gate key to measurement
        results. Measurement results are stored in a 2-dimensional
        numpy array, the first dimension corresponding to the repetition
        and the second to the actual boolean measurement results (ordered
        by the qubits being measured.)
    """
    param_resolver = param_resolver or study.ParamResolver({})
    solved_circuit = protocols.resolve_parameters(circuit, param_resolver)

    return self._sample_measurement_ops(solved_circuit, repetitions)

  def _sample_measurement_ops(
    self,
    program: circuits.Circuit,
    repetitions: int = 1,
  ) -> Dict[str, np.ndarray]:
    """Samples from the circuit at all measurement gates.

    All MeasurementGates must be terminal.
    Note that this does not collapse the wave function.

    Args:
        program: The circuit to sample from.
        repetitions: The number of samples to take.

    Returns:
        A dictionary from measurement gate key to measurement
        results. Measurement results are stored in a 2-dimensional
        numpy array, the first dimension corresponding to the repetition
        and the second to the actual boolean measurement results (ordered
        by the qubits being measured.)
    Raises:
        NotImplementedError: If there are non-terminal measurements in the
            circuit.
        ValueError: If there are multiple MeasurementGates with the same key,
            or if repetitions is negative.
    """
    if not isinstance(program, qsimc.QSimCircuit):
      program = qsimc.QSimCircuit(program, device=program.device)

    if not program.are_all_measurements_terminal():
      raise NotImplementedError("support for non-terminal measurement is not yet implemented")

    measurement_ops = [
      op for _, op, _ in program.findall_operations_with_gate_type(ops.MeasurementGate)
    ]

    # Computes
    # - the list of qubits to be measured
    # - the start (inclusive) and end (exclusive) indices of each measurement
    # - a mapping from measurement key to measurement gate
    measured_qubits = []  # type: List[ops.Qid]
    bounds = {}  # type: Dict[str, Tuple]
    meas_ops = {}  # type: Dict[str, ops.MeasurementGate]
    current_index = 0
    for op in measurement_ops:
      gate = op.gate
      key = protocols.measurement_key(gate)
      meas_ops[key] = gate
      if key in bounds:
        raise ValueError("Duplicate MeasurementGate with key {}".format(key))
      bounds[key] = (current_index, current_index + len(op.qubits))
      measured_qubits.extend(op.qubits)
      current_index += len(op.qubits)

    # Set qsim options
    options = {}
    options.update(self.qsim_options)
    options['c'] = program.translate_cirq_to_qsim(ops.QubitOrder.DEFAULT)

    # Compute indices of measured qubits
    ordered_qubits = ops.QubitOrder.DEFAULT.order_for(program.all_qubits())
    ordered_qubits = list(reversed(ordered_qubits))

    qubit_map = {
      qubit: index for index, qubit in enumerate(ordered_qubits)
    }

    # Simulate
    qsim_state = qsim.qsim_simulate_fullstate(options)
    assert qsim_state.dtype == np.float32
    assert qsim_state.ndim == 1
    final_state = QSimSimulatorState(qsim_state, qubit_map)
    state_vector = final_state.state_vector

    # Measure
    indices = [qubit_map[qubit] for qubit in measured_qubits]
    print(state_vector, indices)


    def _sample_state_vector(state: np.ndarray,
                             measure_indices: List[int],
                             prng_key: np.array,
                             repetitions: int = 1) -> Tuple[np.ndarray, Any]:
      """Helper function to sample from the given state."""

      if repetitions < 0:
        raise ValueError(
            "Number of repetitions cannot be negative. Was {}".format(repetitions))
      num_qubits = int(state.size).bit_length() - 1
      state_shape = (2,) * num_qubits

      if repetitions == 0 or not measure_indices:
        return np.zeros(shape=(repetitions, len(measure_indices)), dtype=np.uint8)

      # Calculate the measurement probabilities.
      probs = _probs(state, measure_indices, num_qubits)

      # Sample over the probability distribution.
      result, new_prng_key = _choice(prng_key, repetitions, probs)
      # Convert to individual qubit measurements.
      return np.asarray([
          value.big_endian_int_to_digits(sample, base=state_shape)
          for sample in result
      ], dtype=np.uint8), new_prng_key


    def _choice(prng_key, repetitions, probabilities):
      """Replacement for np.random.choice()."""
      new_prng_key, *subkeys = jax.random.split(prng_key, repetitions + 1)
      samples = [jax.random.uniform(k) for k in subkeys]

      # Build CDF
      buckets = np.cumsum(probabilities)
      buckets /= buckets[-1]  # For normalization

      # Find the state that matches the uniform sample.
      results = []
      for sample in samples:
        results.append(np.sum(buckets <= sample))
      return results, new_prng_key

    def _probs(state: np.ndarray, indices: List[int],
               num_qubits: int) -> np.ndarray:
      """Returns the probabilities for a measurement on the given indices."""
      shape = (2,) * num_qubits
      tensor = np.reshape(state, shape)
      # Calculate the probabilities for measuring the particular results.
      if len(indices) == num_qubits:
        # We're measuring every qubit, so no need for fancy indexing
        probs = np.abs(tensor)**2
        probs = np.transpose(probs, indices)
        probs = np.reshape(probs, int(np.prod(probs.shape)))
      else:
        # Fancy indexing required
        probs = []
        meas_shape = tuple(shape[i] for i in indices)
        for b in range(np.prod(meas_shape, dtype=int)):
          tensor_ind = linalg.slice_for_qubits_equal_to(
              indices, num_qubits=num_qubits, big_endian_qureg_value=b)
          probs.append(tensor[tensor_ind])
        probs = np.abs(probs)**2
        probs = np.sum(probs, axis=tuple(range(1, len(probs.shape))))

      # To deal with rounding issues, ensure that the probabilities sum to 1.
      probs /= np.sum(probs)
      return probs

    indexed_sample, _prng_key = _sample_state_vector(
        state_vector,
        indices,
        _prng_key,
        repetitions,
    )

    # Applies invert masks of all measurement gates.
    results = {}
    for k, (s, e) in bounds.items():
      before_invert_mask = indexed_sample[:, s:e]
      results[k] = before_invert_mask ^ (
          np.logical_and(before_invert_mask < 2,
                         meas_ops[k].full_invert_mask()))
    return results

    # trial_results = QSimSimulatorTrialResult(params={},
    #                                          measurements=meas_ops,
    #                                          final_simulator_state=final_state)
    # return trial_result.measurements

  def compute_amplitudes_sweep(
      self,
      program: circuits.Circuit,
      bitstrings: Sequence[int],
      params: study.Sweepable,
      qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
  ) -> Sequence[Sequence[complex]]:
    """Computes the desired amplitudes using qsim.

      The initial state is assumed to be the all zeros state.

      Args:
          program: The circuit to simulate.
          bitstrings: The bitstrings whose amplitudes are desired, input as an
            string array where each string is formed from measured qubit values
            according to `qubit_order` from most to least significant qubit,
            i.e. in big-endian ordering.
          param_resolver: Parameters to run with the program.
          qubit_order: Determines the canonical ordering of the qubits. This is
            often used in specifying the initial state, i.e. the ordering of the
            computational basis states.

      Returns:
          List of amplitudes.
      """
    if not isinstance(program, qsimc.QSimCircuit):
      program = qsimc.QSimCircuit(program, device=program.device)

    n_qubits = len(program.all_qubits())
    # qsim numbers qubits in reverse order from cirq
    bitstrings = [format(bitstring, 'b').zfill(n_qubits)[::-1]
                  for bitstring in bitstrings]

    options = {'i': '\n'.join(bitstrings)}
    options.update(self.qsim_options)

    param_resolvers = study.to_resolvers(params)

    trials_results = []
    for prs in param_resolvers:

      solved_circuit = protocols.resolve_parameters(program, prs)

      options['c'] = solved_circuit.translate_cirq_to_qsim(qubit_order)

      amplitudes = qsim.qsim_simulate(options)
      trials_results.append(amplitudes)

    return trials_results

  def simulate_sweep(
      self,
      program: circuits.Circuit,
      params: study.Sweepable,
      qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
      initial_state: Any = None,
  ) -> List['SimulationTrialResult']:
    """Simulates the supplied Circuit.

      This method returns a result which allows access to the entire
      state vector. In contrast to simulate, this allows for sweeping
      over different parameter values.

      Args:
          program: The circuit to simulate.
          params: Parameters to run with the program.
          qubit_order: Determines the canonical ordering of the qubits. This is
            often used in specifying the initial state, i.e. the ordering of the
            computational basis states.
          initial_state: The initial state for the simulation. The form of this
            state depends on the simulation implementation.  See documentation
            of the implementing class for details.

      Returns:
          List of SimulationTrialResults for this run, one for each
          possible parameter resolver.
      """
    if not isinstance(program, qsimc.QSimCircuit):
      program = qsimc.QSimCircuit(program, device=program.device)

    options = {}
    options.update(self.qsim_options)

    param_resolvers = study.to_resolvers(params)

    trials_results = []
    for prs in param_resolvers:
      solved_circuit = protocols.resolve_parameters(program, prs)

      options['c'] = solved_circuit.translate_cirq_to_qsim(qubit_order)
      ordered_qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(
        solved_circuit.all_qubits())
      # qsim numbers qubits in reverse order from cirq
      ordered_qubits = list(reversed(ordered_qubits))

      qubit_map = {
        qubit: index for index, qubit in enumerate(ordered_qubits)
      }

      qsim_state = qsim.qsim_simulate_fullstate(options)
      assert qsim_state.dtype == np.float32
      assert qsim_state.ndim == 1
      final_state = QSimSimulatorState(qsim_state, qubit_map)
      # create result for this parameter
      # TODO: We need to support measurements.
      result = QSimSimulatorTrialResult(params=prs,
                                        measurements={},
                                        final_simulator_state=final_state)
      trials_results.append(result)

    return trials_results
