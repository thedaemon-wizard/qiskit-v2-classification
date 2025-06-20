import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import z_feature_map, real_amplitudes
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Qiskit v2.0.0の新しいprimitives V2
from qiskit_aer.primitives import SamplerV2, EstimatorV2
from qiskit.quantum_info import SparsePauliOp

# IBM Quantum用のインポート
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Session
    from qiskit_ibm_runtime import SamplerV2 as RuntimeSamplerV2
    from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimatorV2
    IBM_RUNTIME_AVAILABLE = True
except ImportError:
    IBM_RUNTIME_AVAILABLE = False
    print("qiskit-ibm-runtime not installed. Install with: pip install qiskit-ibm-runtime")

# 1. データの準備
def generate_sample_data(n_samples=100):
    """簡単な2クラス分類用のサンプルデータを生成"""
    np.random.seed(42)
    
    # クラス0: 中心(-1, -1)周辺のデータ
    class0 = np.random.randn(n_samples//2, 2) * 0.5 + np.array([-1, -1])
    
    # クラス1: 中心(1, 1)周辺のデータ
    class1 = np.random.randn(n_samples//2, 2) * 0.5 + np.array([1, 1])
    
    X = np.vstack([class0, class1])
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
    
    return X, y

# 2. カスタムVQC実装（Qiskit v2.1.0 SamplerV2対応）
class QuantumClassifier:
    def __init__(self, num_features=2, reps=2, backend=None, actual_backend=None):
        self.num_features = num_features
        self.reps = reps
        self.backend = backend
        self.actual_backend = actual_backend  # トランスパイル用のbackendオブジェクト
        
        # 量子回路の構築
        # 実機では常にハードウェア効率的な回路を使用
        use_hardware_circuit = (
            actual_backend is not None or 
            (backend is not None and hasattr(backend, '_backend')) or
            (backend is not None and hasattr(backend, 'backend')) or
            (backend is not None and hasattr(backend, '_mode'))  # Session内のSampler
        )
        
        if use_hardware_circuit:
            print("Building hardware-efficient circuit...")
            self.circuit = self._build_hardware_efficient_circuit(num_features, reps)
        else:
            # ローカルシミュレータの場合のみ標準的な回路を使用
            print("Building circuit for simulator...")
            self.feature_map = z_feature_map(feature_dimension=num_features, reps=1, parameter_prefix='x')
            self.ansatz = real_amplitudes(num_qubits=num_features, reps=reps, parameter_prefix='θ')
            self.circuit = self.feature_map.compose(self.ansatz)
        
        # 測定付き回路
        self.circuit_measured = self.circuit.copy()
        self.circuit_measured.measure_all()
        
        # パラメータ
        self.feature_params = [p for p in self.circuit.parameters if p.name.startswith('x')]
        self.weight_params = [p for p in self.circuit.parameters if p.name.startswith('θ')]
        
        # パラメータを順序付けして保存
        self.ordered_feature_params = sorted(self.feature_params, key=lambda p: p.name)
        self.ordered_weight_params = sorted(self.weight_params, key=lambda p: p.name)
        self.ordered_params = self.ordered_feature_params + self.ordered_weight_params
        
        self.weights = None
        
        # SamplerV2（ローカルまたはランタイム）
        if backend is None:
            self.sampler = SamplerV2()
        else:
            self.sampler = backend
            
        # トランスパイラの設定（必須）
        self.pm = None
        if actual_backend is not None:
            # 実際のbackendオブジェクトがある場合、必ずトランスパイラを設定
            try:
                self.pm = generate_preset_pass_manager(
                    optimization_level=1,
                    backend=actual_backend
                )
                print(f"Transpiler configured for backend: {actual_backend.name}")
            except Exception as e:
                print(f"Warning: Could not create transpiler pass manager: {e}")
                # フォールバックとしてtranspile関数を使用するフラグ
                self.use_transpile = True
        else:
            # デバッグ情報を表示
            print(f"\nCircuit construction details:")
            print(f"- Number of qubits: {self.num_features}")
            print(f"- Number of parameters: {len(self.circuit.parameters)}")
            print(f"- Feature parameters: {len(self.ordered_feature_params)}")
            print(f"- Weight parameters: {len(self.ordered_weight_params)}")
            print(f"- Circuit type: {'Hardware-efficient (no H gates)' if use_hardware_circuit else 'Standard (with H gates)'}")
        if actual_backend is not None:
            print(f"- Backend: {actual_backend.name if hasattr(actual_backend, 'name') else 'provided'}")
        elif backend is not None:
            print(f"- Using runtime sampler")
    
    def _build_hardware_efficient_circuit(self, num_features, reps):
        """実機用のハードウェア効率的な回路を構築"""
        qc = QuantumCircuit(num_features)
        
        # 特徴マップ層（RYゲートのみ使用、Hゲートは使用しない）
        for i in range(num_features):
            param = Parameter(f'x[{i}]')
            qc.ry(2.0 * param, i)  # スケーリング
        
        # 変分層
        param_counter = 0
        for r in range(reps):
            # 回転層
            for i in range(num_features):
                theta = Parameter(f'θ[{param_counter}]')
                qc.ry(theta, i)
                param_counter += 1
            
            # エンタングリング層（線形接続）
            if num_features > 1:
                for i in range(num_features - 1):
                    qc.cx(i, i + 1)
                # 循環接続（最後のキュービットを最初に接続）
                if num_features > 2 and r < reps - 1:
                    qc.cx(num_features - 1, 0)
        
        # 最終回転層
        for i in range(num_features):
            theta = Parameter(f'θ[{param_counter}]')
            qc.ry(theta, i)
            param_counter += 1
        
        print(f"Hardware-efficient circuit created with {len(qc.parameters)} parameters")
        print(f"Circuit uses only RY and CX gates (no H gates)")
        
        return qc
    
    def _compute_probabilities_v2(self, X, weights):
        """SamplerV2を使用して確率を計算（V1 fallbackなしで確実にSession内でV2を使う）"""
        parameter_values = []
        for x in X:
            param_vals = np.concatenate([x, weights])
            parameter_values.append(param_vals)
        parameter_values = np.array(parameter_values)

        circuit = self.circuit_measured
        if self.actual_backend is not None:
            from qiskit import transpile
            circuit = transpile(circuit, backend=self.actual_backend, optimization_level=1)

        job = self.sampler.run([(circuit, parameter_values)])
        result = job.result()
        probabilities = []
        pub_result = result[0]
        data_bin = pub_result.data
        for idx in range(len(X)):
            counts = data_bin.meas.get_counts(idx)
            total_counts = sum(counts.values())
            class1_counts = sum(count for bitstring, count in counts.items()
                                if bitstring.count('1') % 2 == 1)
            prob_class1 = class1_counts / total_counts if total_counts > 0 else 0.5
            probabilities.append(prob_class1)
        return np.array(probabilities)

    
    def _cost_function(self, weights, X, y):
        """コスト関数（交差エントロピー）"""
        try:
            probs = self._compute_probabilities_v2(X, weights)
            
            # 交差エントロピー損失
            epsilon = 1e-10
            loss = -np.mean(y * np.log(probs + epsilon) + 
                           (1 - y) * np.log(1 - probs + epsilon))
            return loss
        except Exception as e:
            print(f"Error in cost function: {e}")
            return 1.0  # エラー時は大きな損失を返す
    
    def fit(self, X, y, max_iter=30):
        """モデルの訓練"""
        # デバッグ：回路の内容を確認
        if hasattr(self.circuit, 'count_ops'):
            ops = self.circuit.count_ops()
            print(f"Circuit operations: {ops}")
            if 'h' in ops:
                print("WARNING: Circuit contains H gates!")
        
        # 初期パラメータ
        initial_weights = np.random.randn(len(self.ordered_weight_params)) * 0.1
        
        print("Training quantum classifier with SamplerV2...")
        # IBM Quantum実機利用時はバッチサイズを小さく！
        batch_size = 1 if self.actual_backend is not None else min(10, len(X))
        
        def batched_cost(weights):
            total_loss = 0
            n_batches = 0
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                total_loss += self._cost_function(weights, batch_X, batch_y)
                n_batches += 1
            return total_loss / n_batches if n_batches > 0 else 0
        
        result = minimize(
            fun=batched_cost,
            x0=initial_weights,
            method='COBYLA',
            options={'maxiter': max_iter, 'disp': True}
        )
        
        self.weights = result.x
        print(f"Training completed. Final loss: {result.fun:.4f}")
        return self

    
    def predict_proba(self, X):
        """確率予測"""
        probs_class1 = self._compute_probabilities_v2(X, self.weights)
        probs_class0 = 1 - probs_class1
        return np.column_stack([probs_class0, probs_class1])
    
    def predict(self, X):
        """クラス予測"""
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)
    
    def score(self, X, y):
        """精度スコア"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

# 3. EstimatorV2を使用した代替実装
class QuantumClassifierEstimator:
    """EstimatorV2を使用した量子分類器（期待値ベース）"""
    def __init__(self, num_features=2, reps=2, backend=None, actual_backend=None):
        self.num_features = num_features
        self.reps = reps
        self.actual_backend = actual_backend
        
        # 量子回路の構築（新しい関数を使用）
        self.feature_map = z_feature_map(feature_dimension=num_features, reps=1, parameter_prefix='x')
        self.ansatz = real_amplitudes(num_qubits=num_features, reps=reps, parameter_prefix='θ')
        self.circuit = self.feature_map.compose(self.ansatz)
        
        # パラメータ
        self.feature_params = [p for p in self.circuit.parameters if p.name.startswith('x')]
        self.weight_params = [p for p in self.circuit.parameters if p.name.startswith('θ')]
        
        # パラメータを順序付けして保存
        self.ordered_feature_params = sorted(self.feature_params, key=lambda p: p.name)
        self.ordered_weight_params = sorted(self.weight_params, key=lambda p: p.name)
        self.ordered_params = self.ordered_feature_params + self.ordered_weight_params
        
        self.weights = None
        
        # EstimatorV2
        if backend is None:
            self.estimator = EstimatorV2()
        else:
            self.estimator = backend
        
        # 観測量（Z演算子）
        self.observable = SparsePauliOp.from_list([("Z" * num_features, 1.0)])
    
    def _compute_expectations(self, X, weights):
        """期待値を計算"""
        # パラメータ値の2D配列を作成
        parameter_values = []
        for x in X:
            param_vals = np.concatenate([x, weights])
            parameter_values.append(param_vals)
        
        parameter_values = np.array(parameter_values)
        
        # 実行
        job = self.estimator.run([(self.circuit, self.observable, parameter_values)])
        result = job.result()
        
        # 期待値を取得
        expectations = []
        pub_result = result[0]
        for idx in range(len(X)):
            exp_val = pub_result.data.evs[idx]
            expectations.append(exp_val)
        
        return np.array(expectations)
    
    def fit(self, X, y, max_iter=30):
        """モデルの訓練"""
        # 初期パラメータ
        initial_weights = np.random.randn(len(self.ordered_weight_params)) * 0.1
        
        def cost_function(weights):
            expectations = self._compute_expectations(X, weights)
            # 期待値を確率に変換
            probs = (expectations + 1) / 2
            
            # 交差エントロピー損失
            epsilon = 1e-10
            loss = -np.mean(y * np.log(probs + epsilon) + 
                           (1 - y) * np.log(1 - probs + epsilon))
            return loss
        
        print("Training quantum classifier with EstimatorV2...")
        result = minimize(
            fun=cost_function,
            x0=initial_weights,
            method='COBYLA',
            options={'maxiter': max_iter, 'disp': True}
        )
        
        self.weights = result.x
        print(f"Training completed. Final loss: {result.fun:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """期待値から確率に変換"""
        expectations = self._compute_expectations(X, self.weights)
        # 期待値を[0,1]の範囲に正規化
        probs_class1 = (expectations + 1) / 2
        probs_class0 = 1 - probs_class1
        return np.column_stack([probs_class0, probs_class1])
    
    def predict(self, X):
        """クラス予測"""
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)
    
    def score(self, X, y):
        """精度スコア"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

# 4. ローカルシミュレータでの実行
def run_on_simulator():
    """ローカルシミュレータで量子分類器を実行"""
    # データの準備
    X, y = generate_sample_data(80)  # データ量を減らす
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # データの正規化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 量子分類器の作成と訓練
    print("Using SamplerV2 for classification...")
    qc = QuantumClassifier(num_features=2, reps=2)
    qc.fit(X_train_scaled, y_train, max_iter=30)
    
    # 評価
    train_score = qc.score(X_train_scaled, y_train)
    test_score = qc.score(X_test_scaled, y_test)
    
    print(f"\nTraining accuracy: {train_score:.3f}")
    print(f"Testing accuracy: {test_score:.3f}")
    
    return qc

# 5. EstimatorV2での実行
def run_with_estimator():
    """EstimatorV2を使用した量子分類器を実行"""
    # データの準備
    X, y = generate_sample_data(60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # データの正規化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # EstimatorV2を使用した量子分類器
    print("\nUsing EstimatorV2 for classification...")
    qc_est = QuantumClassifierEstimator(num_features=2, reps=2)
    qc_est.fit(X_train_scaled, y_train, max_iter=30)
    
    # 評価
    train_score = qc_est.score(X_train_scaled, y_train)
    test_score = qc_est.score(X_test_scaled, y_test)
    
    print(f"\nTraining accuracy: {train_score:.3f}")
    print(f"Testing accuracy: {test_score:.3f}")
    
    return qc_est

# 6. IBM Quantum実機での実行
def run_on_ibm_quantum(api_token=None, use_simulator=False):
    """IBM Quantum実機で実行（Qiskit Runtime V2専用, サンプル関数省略なし）"""
    # 1. サービスの初期化
    try:
        if api_token:
            service = QiskitRuntimeService(channel="ibm_cloud", token=api_token)
        else:
            service = QiskitRuntimeService(channel="ibm_cloud")
    except Exception as e:
        print(f"Error: {e}")
        print("Please save your IBM Quantum account first:")
        print("QiskitRuntimeService.save_account(channel='ibm_cloud', token='YOUR_TOKEN')")
        return None

    # 2. バックエンド取得
    try:
        if use_simulator:
            backend = service.least_busy(operational=True, simulator=True)
        else:
            backends = service.backends(simulator=False, operational=True)
            suitable_backends = [b for b in backends if b.configuration().n_qubits >= 2]
            if not suitable_backends:
                print("No suitable quantum backends available.")
                return None
            backend = min(suitable_backends, key=lambda b: b.status().pending_jobs)
        print(f"Using backend: {backend.name}")
    except Exception as e:
        print(f"Error retrieving backend: {e}")
        return None

    # 3. データの準備
    X, y = generate_sample_data(20)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Session内で全処理
    try:
        with Session(backend=backend) as session:
            sampler = SamplerV2(session=session)
            print("Using SamplerV2 inside active session")
            qc = QuantumClassifier(num_features=2, reps=1, backend=sampler, actual_backend=backend)
            qc.fit(X_train_scaled, y_train, max_iter=10)
            train_score = qc.score(X_train_scaled, y_train)
            test_score = qc.score(X_test_scaled, y_test)
            print(f"\nResults from {backend.name}:")
            print(f"Training accuracy: {train_score:.3f}")
            print(f"Testing accuracy: {test_score:.3f}")
    except Exception as e:
        print(f"Error running jobs on quantum hardware: {e}")
        return None

    print("\nSession closed. All jobs completed.")
    return qc

# 8. シンプルな量子回路の可視化
def run_on_ibm_quantum_simple(use_simulator=False):
    """IBM Quantum実機で実行（シンプル版）"""
    if not IBM_RUNTIME_AVAILABLE:
        print("qiskit-ibm-runtime is not installed.")
        return None
    
    try:
        # サービスの初期化
        service = QiskitRuntimeService(channel="ibm_quantum")
        
        # バックエンドの選択
        if use_simulator:
            backend = service.least_busy(simulator=True)
        else:
            backend = service.least_busy(simulator=False, min_num_qubits=2)
        
        print(f"Using backend: {backend.name}")
        
        # データの準備
        X, y = generate_sample_data(20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Sampler V1を使用（より安定）
        from qiskit_ibm_runtime import Sampler
        sampler = Sampler(backend=backend)
        sampler.options.execution.shots = 1024
        
        # 簡単な量子回路でテスト
        qc = QuantumClassifier(num_features=2, reps=1, backend=sampler, actual_backend=backend)
        qc.fit(X_train_scaled, y_train, max_iter=5)
        
        train_score = qc.score(X_train_scaled, y_train)
        test_score = qc.score(X_test_scaled, y_test)
        
        print(f"\nResults:")
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Testing accuracy: {test_score:.3f}")
        
        return qc
        
    except Exception as e:
        print(f"Error: {e}")
        return None
def visualize_quantum_circuit():
    """量子分類器の回路を可視化"""
    # 特徴マップ（新しい関数を使用）
    feature_map = z_feature_map(feature_dimension=2, reps=1)
    print("Feature Map Circuit:")
    print(feature_map.draw(fold=80))
    
    # 変分回路（新しい関数を使用）
    ansatz = real_amplitudes(num_qubits=2, reps=2)
    print("\nAnsatz Circuit:")
    print(ansatz.draw(fold=80))
    
    # 完全な回路
    full_circuit = feature_map.compose(ansatz)
    full_circuit.measure_all()
    print("\nComplete Classification Circuit:")
    print(full_circuit.draw(fold=80))

# 11. 使用例
def run_on_ibm_quantum_direct(use_simulator=False):
    """IBM Quantum実機で実行（backend.run()を直接使用）"""
    if not IBM_RUNTIME_AVAILABLE:
        print("qiskit-ibm-runtime is not installed.")
        return None
    
    try:
        # サービスの初期化
        service = QiskitRuntimeService(channel="ibm_quantum")
        
        # バックエンドの選択
        if use_simulator:
            backend = service.least_busy(simulator=True)
        else:
            backend = service.least_busy(simulator=False, min_num_qubits=2)
        
        print(f"Using backend: {backend.name}")
        print(f"Backend version: {backend.version}")
        
        # データの準備
        X, y = generate_sample_data(16)  # さらに少ないデータ
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 簡単なテスト回路を実行
        print("\nTesting backend with simple circuit...")
        test_qc = QuantumCircuit(2)
        test_qc.ry(1.57, 0)  # π/2 rotation (similar to H but using RY)
        test_qc.cx(0, 1)
        test_qc.measure_all()
        
        # トランスパイル
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        test_qc_transpiled = pm.run(test_qc)
        
        # テスト実行
        job = backend.run(test_qc_transpiled, shots=100)
        result = job.result()
        counts = result.get_counts()
        print(f"Test circuit results: {counts}")
        
        # カスタム分類器（backend.run()を使用）
        class DirectQuantumClassifier:
            def __init__(self, backend, num_features=2, reps=1):
                self.backend = backend
                self.num_features = num_features
                self.reps = reps
                
                # 実機用のハードウェア効率的な回路を作成
                self.circuit = self._build_hardware_efficient_circuit(num_features, reps)
                self.circuit.measure_all()
                
                # パラメータ
                self.feature_params = sorted([p for p in self.circuit.parameters if p.name.startswith('x')], 
                                           key=lambda p: p.name)
                self.weight_params = sorted([p for p in self.circuit.parameters if p.name.startswith('θ')], 
                                          key=lambda p: p.name)
                self.weights = None
                
                # トランスパイラ
                self.pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
            
            def _build_hardware_efficient_circuit(self, num_features, reps):
                """実機用の回路（Hゲートなし）"""
                qc = QuantumCircuit(num_features)
                
                # 特徴マップ層（RYゲートのみ）
                for i in range(num_features):
                    param = Parameter(f'x[{i}]')
                    qc.ry(2.0 * param, i)
                
                # 変分層
                param_counter = 0
                for r in range(reps):
                    for i in range(num_features):
                        theta = Parameter(f'θ[{param_counter}]')
                        qc.ry(theta, i)
                        param_counter += 1
                    
                    if num_features > 1:
                        for i in range(num_features - 1):
                            qc.cx(i, i + 1)
                
                # 最終層
                for i in range(num_features):
                    theta = Parameter(f'θ[{param_counter}]')
                    qc.ry(theta, i)
                    param_counter += 1
                
                return qc
            
            def _run_circuits(self, X, weights):
                """backend.run()を使用して回路を実行"""
                circuits = []
                for x in X:
                    param_dict = {}
                    for i, p in enumerate(self.feature_params):
                        param_dict[p] = float(x[i])
                    for i, p in enumerate(self.weight_params):
                        param_dict[p] = float(weights[i])
                    bound_circuit = self.circuit.assign_parameters(param_dict)
                    transpiled = self.pm.run(bound_circuit)
                    circuits.append(transpiled)
                
                # バッチ実行
                job = self.backend.run(circuits, shots=1024)
                result = job.result()
                
                # 確率を計算
                probabilities = []
                for i in range(len(circuits)):
                    counts = result.get_counts(i)
                    total = sum(counts.values())
                    class1_counts = sum(count for bitstring, count in counts.items()
                                      if bitstring.count('1') % 2 == 1)
                    prob = class1_counts / total if total > 0 else 0.5
                    probabilities.append(prob)
                
                return np.array(probabilities)
            
            def fit(self, X, y, max_iter=5):
                """簡単な最適化"""
                initial_weights = np.random.randn(len(self.weight_params)) * 0.1
                
                def cost(weights):
                    probs = self._run_circuits(X, weights)
                    epsilon = 1e-10
                    loss = -np.mean(y * np.log(probs + epsilon) + 
                                   (1 - y) * np.log(1 - probs + epsilon))
                    print(f"Loss: {loss:.4f}")
                    return loss
                
                print("Training with direct backend execution...")
                result = minimize(cost, initial_weights, method='COBYLA', 
                                options={'maxiter': max_iter})
                self.weights = result.x
                return self
            
            def predict(self, X):
                probs = self._run_circuits(X, self.weights)
                return (probs > 0.5).astype(int)
            
            def score(self, X, y):
                predictions = self.predict(X)
                return np.mean(predictions == y)
        
        # 分類器の作成と訓練
        print("\nTraining quantum classifier with direct backend execution...")
        qc = DirectQuantumClassifier(backend, num_features=2, reps=1)
        qc.fit(X_train_scaled, y_train, max_iter=3)
        
        # 評価
        print("\nEvaluating...")
        train_score = qc.score(X_train_scaled, y_train)
        test_score = qc.score(X_test_scaled, y_test)
        
        print(f"\nResults from {backend.name}:")
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Testing accuracy: {test_score:.3f}")
        
        return qc
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None
if __name__ == "__main__":
    print("=== Quantum Classification with Qiskit v2.1.0 (SamplerV2) ===\n")
    
    # Qiskitバージョンの確認
    import qiskit
    print(f"Qiskit version: {qiskit.__version__}\n")
    
    # 回路の可視化
    print("1. Visualizing Quantum Circuits:")
    visualize_quantum_circuit()
    
    # ローカルシミュレータでの実行（SamplerV2）
    print("\n2. Running on Local Simulator with SamplerV2:")
    qc_local = run_on_simulator()
    
    # EstimatorV2での実行
    print("\n3. Running with EstimatorV2:")
    qc_estimator = run_with_estimator()
    
    # IBM Quantumでの実行方法
    print("\n4. IBM Quantum Execution:")
    print("To run on IBM Quantum real hardware:")
    print("a) Install qiskit-ibm-runtime: pip install qiskit-ibm-runtime")
    print("b) Create an account at https://quantum.ibm.com")
    print("c) Save your credentials (do this once):")
    print("   from qiskit_ibm_runtime import QiskitRuntimeService")
    print("   # For new accounts (recommended):")
    print("   QiskitRuntimeService.save_account(channel='ibm_cloud', token='YOUR_TOKEN')")
    print("   # For existing accounts:")
    print("   QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_TOKEN')")
    print("d) After saving, you can run:")
    print("   - Direct backend (most stable): qc_ibm = run_on_ibm_quantum_direct()")
    print("   - Simple version: qc_ibm = run_on_ibm_quantum_simple()")
    print("   - Full version (fixed): qc_ibm = run_on_ibm_quantum()")
    print("   - With simulator: add use_simulator=True to any method")
    print("\nNote: Real quantum hardware may have queue times!")
    print("\nImportant for real hardware:")
    print("- All circuits must be transpiled to native gates (rz, sx, x, cx)")
    print("- The fixed version now properly handles transpilation")
    print("- Expect lower accuracy due to quantum noise")
    print("\nTroubleshooting:")
    print("- Use ibm_cloud channel for new accounts")
    print("- The ibm_quantum channel is deprecated (sunset July 1)")
    print("- If you get gate errors, the fixed version handles this")
    
    # 実際に実行する場合はコメントを外す
    # qc_ibm = run_on_ibm_quantum_direct()  # 最も安定（推奨）
    # qc_ibm = run_on_ibm_quantum_simple()  # 簡易版
    QiskitRuntimeService.save_account(channel='ibm_cloud', overwrite=True, token='YOUR_TOKEN')

    qc_ibm = run_on_ibm_quantum()  # フル機能版