import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_mjpeg/flutter_mjpeg.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:shared_preferences/shared_preferences.dart';

const Color ink = Color(0xFF15233B);
const Color mutedInk = Color(0xFF6B7890);
const Color canvas = Color(0xFFF4F7FB);
const Color teal = Color(0xFF147D78);
const Color paleTeal = Color(0xFFE4F5F2);
const Color amber = Color(0xFFB87512);
const Color paleAmber = Color(0xFFFFF2DB);
const Color red = Color(0xFFC2414A);
const Color paleRed = Color(0xFFFFE9EA);
const Color line = Color(0xFFE5EAF1);

void main() => runApp(const FallDetectionApp());

class FallDetectionApp extends StatelessWidget {
  const FallDetectionApp({super.key});

  @override
  Widget build(BuildContext context) {
    final scheme = ColorScheme.fromSeed(seedColor: teal, brightness: Brightness.light);
    return MaterialApp(
      title: 'FallGuard AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: scheme,
        scaffoldBackgroundColor: canvas,
        fontFamily: 'Roboto',
        appBarTheme: const AppBarTheme(backgroundColor: canvas, foregroundColor: ink, elevation: 0),
        cardTheme: CardThemeData(
          color: Colors.white,
          elevation: 0,
          margin: EdgeInsets.zero,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(22), side: const BorderSide(color: line)),
        ),
      ),
      home: const MonitorShell(),
    );
  }
}

class MonitorShell extends StatefulWidget {
  const MonitorShell({super.key});

  @override
  State<MonitorShell> createState() => _MonitorShellState();
}

class _MonitorShellState extends State<MonitorShell> {
  int _tab = 0;
  static const String _serverUrlPreferenceKey = 'fallguard_server_url';
  String _serverUrl = 'http://192.168.1.10:8000';
  bool _preferencesLoaded = false;
  bool _hasSavedServerUrl = false;
  bool _apiOnline = false;
  bool _checking = false;
  String _connectionMessage = 'Chưa kiểm tra kết nối';
  Map<String, dynamic>? _apiStatus;
  int _streamGeneration = 0;
  Timer? _statusTimer;

  @override
  void initState() {
    super.initState();
    _restoreSavedServerUrl();
    _statusTimer = Timer.periodic(const Duration(seconds: 5), (_) {
      if (mounted && _preferencesLoaded && _hasSavedServerUrl && !_checking) {
        _refreshStatusSilently();
      }
    });
  }

  Future<void> _restoreSavedServerUrl() async {
    String? savedUrl;
    try {
      final prefs = await SharedPreferences.getInstance();
      savedUrl = prefs.getString(_serverUrlPreferenceKey);
    } catch (_) {
      // App vẫn hoạt động nếu đọc cài đặt lưu trữ gặp lỗi.
    }

    final normalized = savedUrl?.trim().replaceAll(RegExp(r'/+$'), '');
    if (!mounted) return;
    setState(() {
      _preferencesLoaded = true;
      if (normalized != null && normalized.isNotEmpty) {
        _serverUrl = normalized;
        _hasSavedServerUrl = true;
      } else {
        _connectionMessage = 'Nhập địa chỉ laptop trong mục Cài đặt';
      }
    });

    // Nếu đã lưu địa chỉ từ lần trước, tự kiểm tra kết nối khi mở app.
    if (normalized != null && normalized.isNotEmpty) {
      await _checkConnection();
    }
  }

  Future<void> _persistServerUrl(String value) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(_serverUrlPreferenceKey, value);
    } catch (_) {
      if (mounted) {
        setState(() {
          _connectionMessage = 'Không thể lưu địa chỉ trên điện thoại';
        });
      }
    }
  }

  @override
  void dispose() {
    _statusTimer?.cancel();
    super.dispose();
  }

  Future<void> _refreshStatusSilently() async {
    try {
      final base = _serverUrl.replaceAll(RegExp(r'/+$'), '');
      final response = await http.get(Uri.parse('$base/api/status')).timeout(const Duration(seconds: 4));
      if (response.statusCode < 200 || response.statusCode >= 300) throw Exception('HTTP ${response.statusCode}');
      final decoded = jsonDecode(response.body);
      _acceptStatus(decoded);
    } catch (_) {
      if (!mounted) return;
      setState(() {
        _apiOnline = false;
        _apiStatus = null;
        _connectionMessage = 'Chưa kết nối được backend';
      });
    }
  }

  void _acceptStatus(dynamic decoded) {
    if (!mounted) return;
    final next = decoded is Map<String, dynamic> ? decoded : <String, dynamic>{'response': decoded};
    final previous = _apiStatus;
    final previousUptime = num.tryParse(previous?['uptime_seconds']?.toString() ?? '');
    final nextUptime = num.tryParse(next['uptime_seconds']?.toString() ?? '');
    final uptimeReset = previousUptime != null && nextUptime != null && nextUptime + 2 < previousUptime;
    final cameraCameBack = previous?['camera'] != null &&
        previous?['camera']?.toString() != 'online' && next['camera']?.toString() == 'online';
    final backendReturned = !_apiOnline;
    setState(() {
      if (uptimeReset || cameraCameBack || backendReturned) {
        // Changing the key/URL disposes the old MJPEG request and opens a fresh one.
        _streamGeneration++;
      }
      _apiOnline = true;
      _apiStatus = next;
      _connectionMessage = 'Đã kết nối API';
    });
  }

  Future<void> _checkConnection() async {
    final urlToCheck = _serverUrl.trim().replaceAll(RegExp(r'/+$'), '');
    if (urlToCheck.isEmpty) {
      if (mounted) {
        setState(() => _connectionMessage = 'Vui lòng nhập địa chỉ laptop');
      }
      return;
    }
    setState(() {
      _checking = true;
      _connectionMessage = 'Đang kiểm tra API...';
    });
    try {
      final uri = Uri.parse('$urlToCheck/api/status');
      final response = await http.get(uri).timeout(const Duration(seconds: 4));
      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw Exception('HTTP ${response.statusCode}');
      }
      final decoded = jsonDecode(response.body);
      await _persistServerUrl(urlToCheck);
      if (!mounted) return;
      setState(() => _hasSavedServerUrl = true);
      _acceptStatus(decoded);
    } catch (e) {
      if (mounted) {
        setState(() {
          _apiOnline = false;
          _apiStatus = null;
          _connectionMessage = 'Chưa kết nối được: ${e.toString().replaceFirst('Exception: ', '')}';
        });
      }
    } finally {
      if (mounted) setState(() => _checking = false);
    }
  }

  void _saveServerUrl(String value) {
    final normalized = value.trim().replaceAll(RegExp(r'/+$'), '');
    if (normalized.isEmpty) {
      setState(() => _connectionMessage = 'Vui lòng nhập địa chỉ laptop');
      return;
    }
    setState(() {
      _serverUrl = normalized;
      _apiOnline = false;
      _apiStatus = null;
      _hasSavedServerUrl = true;
      _streamGeneration++;
      _connectionMessage = 'Đã lưu địa chỉ trên điện thoại';
    });
    // Lưu ngay khi người dùng nhấn "Lưu địa chỉ"; _checkConnection cũng lưu
    // lại khi kết nối thành công, nên không cần nhập lại sau khi mở app lần sau.
    unawaited(_persistServerUrl(normalized));
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Đã lưu địa chỉ backend trên điện thoại.')),
    );
  }

  @override
  Widget build(BuildContext context) {
    final pages = <Widget>[
      OverviewPage(
        serverUrl: _serverUrl,
        apiOnline: _apiOnline,
        apiStatus: _apiStatus,
        connectionMessage: _connectionMessage,
        onOpenCamera: () => setState(() => _tab = 1),
        onCheckConnection: _checkConnection,
        checking: _checking,
      ),
      CameraPage(serverUrl: _serverUrl, apiOnline: _apiOnline, apiStatus: _apiStatus, streamGeneration: _streamGeneration, onCheckConnection: _checkConnection, checking: _checking),
      EventsPage(serverUrl: _serverUrl),
      SettingsPage(
        serverUrl: _serverUrl,
        apiOnline: _apiOnline,
        connectionMessage: _connectionMessage,
        apiStatus: _apiStatus,
        onSaveServerUrl: _saveServerUrl,
        onCheckConnection: _checkConnection,
        checking: _checking,
      ),
    ];

    return Scaffold(
      body: SafeArea(child: IndexedStack(index: _tab, children: pages)),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _tab,
        onDestinationSelected: (i) => setState(() => _tab = i),
        backgroundColor: Colors.white,
        indicatorColor: paleTeal,
        destinations: const [
          NavigationDestination(icon: Icon(Icons.space_dashboard_outlined), selectedIcon: Icon(Icons.space_dashboard), label: 'Tổng quan'),
          NavigationDestination(icon: Icon(Icons.videocam_outlined), selectedIcon: Icon(Icons.videocam), label: 'Camera'),
          NavigationDestination(icon: Icon(Icons.notifications_none), selectedIcon: Icon(Icons.notifications), label: 'Sự kiện'),
          NavigationDestination(icon: Icon(Icons.settings_outlined), selectedIcon: Icon(Icons.settings), label: 'Cài đặt'),
        ],
      ),
    );
  }
}

class OverviewPage extends StatelessWidget {
  final String serverUrl;
  final bool apiOnline;
  final String connectionMessage;
  final Map<String, dynamic>? apiStatus;
  final VoidCallback onOpenCamera;
  final VoidCallback onCheckConnection;
  final bool checking;
  const OverviewPage({super.key, required this.serverUrl, required this.apiOnline, required this.connectionMessage, required this.apiStatus, required this.onOpenCamera, required this.onCheckConnection, required this.checking});

  @override
  Widget build(BuildContext context) {
    return ListView(padding: const EdgeInsets.fromLTRB(20, 18, 20, 28), children: [
      const AppHeader(title: 'FallGuard AI', subtitle: 'Giám sát té ngã thông minh'),
      const SizedBox(height: 22),
      Row(children: [
        const Expanded(child: Text('Tổng quan', style: TextStyle(fontSize: 27, fontWeight: FontWeight.w800, color: ink))),
        _pill(apiOnline ? 'BACKEND KẾT NỐI' : 'CHỜ BACKEND', apiOnline ? teal : amber, apiOnline ? paleTeal : paleAmber),
      ]),
      const SizedBox(height: 6),
      const Text('Theo dõi camera laptop từ điện thoại của bạn.', style: TextStyle(color: mutedInk, height: 1.45)),
      const SizedBox(height: 18),
      ConnectionCard(apiOnline: apiOnline, message: connectionMessage, serverUrl: serverUrl, checking: checking, onCheck: onCheckConnection),
      const SizedBox(height: 14),
      Row(children: [
        Expanded(child: MetricCard(icon: Icons.videocam_outlined, label: 'Camera', value: apiStatus?['camera']?.toString() ?? 'Chưa kết nối', hint: 'Nguồn laptop', tint: paleTeal, iconColor: teal)),
        const SizedBox(width: 12),
        Expanded(child: MetricCard(icon: Icons.memory, label: 'AI Engine', value: apiStatus?['ai']?.toString() ?? 'Chờ backend', hint: 'Transformer 8 lớp', tint: const Color(0xFFECEBFF), iconColor: const Color(0xFF6862C8))),
      ]),
      const SizedBox(height: 18),
      SectionTitle(title: 'Xem camera', action: 'Mở', onAction: onOpenCamera),
      const SizedBox(height: 10),
      const CameraPlaceholder(height: 210),
      const SizedBox(height: 10),
      OutlinedButton.icon(onPressed: onOpenCamera, icon: const Icon(Icons.play_circle_outline), label: const Text('Mở trang camera'), style: OutlinedButton.styleFrom(minimumSize: const Size.fromHeight(48), foregroundColor: teal, side: const BorderSide(color: Color(0xFFB9DCD6)), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)))),
      const SizedBox(height: 22),
      const SectionTitle(title: 'Sự kiện gần đây'),
      const SizedBox(height: 10),
      Card(child: Padding(padding: const EdgeInsets.all(14), child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [const Icon(Icons.notifications_active_outlined, color: teal), const SizedBox(width: 10), Expanded(child: Text(apiOnline ? 'Sự kiện thật được lấy từ backend. Mở tab Sự kiện để xem lịch sử cảnh báo.' : 'Kết nối backend laptop để nhận trạng thái AI và sự kiện thật.', style: const TextStyle(color: mutedInk, height: 1.45)))]))),
    ]);
  }
}

class CameraPage extends StatelessWidget {
  final String serverUrl;
  final bool apiOnline;
  final Map<String, dynamic>? apiStatus;
  final int streamGeneration;
  final VoidCallback onCheckConnection;
  final bool checking;
  const CameraPage({super.key, required this.serverUrl, required this.apiOnline, required this.apiStatus, required this.streamGeneration, required this.onCheckConnection, required this.checking});

  Future<void> _controlCamera(BuildContext context, String action) async {
    final base = serverUrl.replaceAll(RegExp(r'/+$'), '');
    try {
      final response = await http.post(Uri.parse('$base/api/camera/$action')).timeout(const Duration(seconds: 5));
      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw Exception('HTTP ${response.statusCode}');
      }
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(action == 'start' ? 'Đã gửi lệnh bật camera.' : 'Đã gửi lệnh dừng camera.')));
      }
      onCheckConnection();
    } catch (e) {
      if (context.mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Không điều khiển được camera: $e')));
    }
  }

  @override
  Widget build(BuildContext context) {
    final cameraState = apiStatus?['camera']?.toString() ?? 'unknown';
    final aiState = apiStatus?['ai']?.toString() ?? 'unknown';
    final lastError = apiStatus?['last_error']?.toString() ?? '';
    final streamUrl = '${serverUrl.replaceAll(RegExp(r'/+$'), '')}/stream.mjpg?generation=$streamGeneration';
    return ListView(padding: const EdgeInsets.fromLTRB(20, 18, 20, 28), children: [
      const AppHeader(title: 'Camera trực tiếp', subtitle: 'Nguồn video AI từ laptop'),
      const SizedBox(height: 20),
      Row(children: [const Expanded(child: Text('Camera 01', style: TextStyle(fontSize: 25, fontWeight: FontWeight.w800, color: ink))), _pill(apiOnline ? 'API ONLINE' : 'OFFLINE', apiOnline ? teal : mutedInk, apiOnline ? paleTeal : const Color(0xFFEDF0F4))]),
      const SizedBox(height: 12),
      ClipRRect(
        borderRadius: BorderRadius.circular(22),
        child: Container(
          color: const Color(0xFF101A2A),
          height: 270,
          width: double.infinity,
          child: apiOnline
              ? Mjpeg(
                  key: ValueKey('$streamUrl-$streamGeneration'),
                  stream: streamUrl,
                  isLive: true,
                  timeout: const Duration(seconds: 10),
                  fit: BoxFit.contain,
                  loading: (_) => const CameraPlaceholder(height: 270, large: true),
                  error: (_, error, stack) => const CameraPlaceholder(height: 270, large: true),
                )
              : const CameraPlaceholder(height: 270, large: true),
        ),
      ),
      const SizedBox(height: 14),
      Card(child: Padding(padding: const EdgeInsets.all(16), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [Icon(apiOnline && cameraState == 'online' ? Icons.check_circle_outline : Icons.info_outline, color: apiOnline && cameraState == 'online' ? teal : amber), const SizedBox(width: 8), Expanded(child: Text(apiOnline ? 'Camera: $cameraState · AI: $aiState' : 'Chưa kết nối backend', style: const TextStyle(fontWeight: FontWeight.w800, color: ink)))]),
        const SizedBox(height: 8),
        Text('Backend: $serverUrl', style: const TextStyle(fontSize: 12, color: mutedInk)),
        if (lastError.isNotEmpty) ...[
          const SizedBox(height: 8),
          Text('Lỗi backend: $lastError', style: const TextStyle(color: red, fontSize: 12, height: 1.4)),
        ],
        const SizedBox(height: 8),
        Text('Thiết bị: ${apiStatus?['device'] ?? '—'} · FPS xử lý: ${apiStatus?['fps'] ?? '—'} · Số người: ${apiStatus?['people_count'] ?? '—'}', style: const TextStyle(fontSize: 12, color: mutedInk)),
      ]))),
      const SizedBox(height: 14),
      Row(children: [
        Expanded(child: FilledButton.icon(onPressed: apiOnline ? () => _controlCamera(context, 'start') : null, icon: const Icon(Icons.play_arrow), label: const Text('Bật camera'), style: FilledButton.styleFrom(backgroundColor: teal, minimumSize: const Size.fromHeight(48), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(13))))),
        const SizedBox(width: 10),
        Expanded(child: OutlinedButton.icon(onPressed: apiOnline ? () => _controlCamera(context, 'stop') : null, icon: const Icon(Icons.stop_circle_outlined), label: const Text('Dừng camera'), style: OutlinedButton.styleFrom(minimumSize: const Size.fromHeight(48), foregroundColor: red, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(13))))),
      ]),
      const SizedBox(height: 10),
      OutlinedButton.icon(onPressed: checking ? null : onCheckConnection, icon: checking ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2, color: teal)) : const Icon(Icons.refresh), label: Text(checking ? 'Đang kiểm tra...' : 'Làm mới trạng thái backend'), style: OutlinedButton.styleFrom(foregroundColor: teal, minimumSize: const Size.fromHeight(46), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)))),
      const SizedBox(height: 12),
      const Row(children: [Icon(Icons.info_outline, size: 16, color: amber), SizedBox(width: 7), Expanded(child: Text('Để xem từ điện thoại, laptop và điện thoại phải cùng Wi-Fi; nhập IPv4 của laptop trong phần Cài đặt.', style: TextStyle(fontSize: 12, color: mutedInk)))])
    ]);
  }
}

class EventsPage extends StatefulWidget {
  final String serverUrl;
  const EventsPage({super.key, required this.serverUrl});
  @override
  State<EventsPage> createState() => _EventsPageState();
}

class _EventsPageState extends State<EventsPage> {
  late Future<List<Map<String, dynamic>>> _eventsFuture;
  Timer? _refreshTimer;
  WebSocketChannel? _channel;
  StreamSubscription<dynamic>? _wsSubscription;

  @override
  void initState() {
    super.initState();
    _eventsFuture = _loadEvents();
    _connectWebSocket();
    // Polling is a fallback in case the WebSocket disconnects.
    _refreshTimer = Timer.periodic(const Duration(seconds: 15), (_) {
      if (mounted) setState(() => _eventsFuture = _loadEvents());
    });
  }

  void _connectWebSocket() {
    _wsSubscription?.cancel();
    try {
      final base = widget.serverUrl.replaceAll(RegExp(r'/+$'), '');
      final wsBase = base.startsWith('https://')
          ? base.replaceFirst('https://', 'wss://')
          : base.replaceFirst('http://', 'ws://');
      final channel = WebSocketChannel.connect(Uri.parse('$wsBase/ws'));
      _channel = channel;
      _wsSubscription = channel.stream.listen((dynamic message) {
        try {
          final payload = jsonDecode(message.toString());
          if (payload is Map && payload['type'] == 'fall_event' && mounted) {
            setState(() => _eventsFuture = _loadEvents());
          }
        } catch (_) {
          // Ignore malformed or non-JSON WebSocket messages.
        }
      }, onError: (_) {}, onDone: () {});
    } catch (_) {
      // REST polling remains available if WebSocket cannot connect.
    }
  }

  @override
  void didUpdateWidget(covariant EventsPage oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.serverUrl != widget.serverUrl) {
      _eventsFuture = _loadEvents();
      _connectWebSocket();
    }
  }

  @override
  void dispose() {
    _refreshTimer?.cancel();
    _wsSubscription?.cancel();
    _channel?.sink.close();
    super.dispose();
  }

  Future<List<Map<String, dynamic>>> _loadEvents() async {
    final base = widget.serverUrl.replaceAll(RegExp(r'/+$'), '');
    final response = await http.get(Uri.parse('$base/api/events?limit=50')).timeout(const Duration(seconds: 5));
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw Exception('API trả về HTTP ${response.statusCode}');
    }
    final decoded = jsonDecode(response.body);
    final items = decoded is Map<String, dynamic> ? decoded['items'] : null;
    if (items is! List) return <Map<String, dynamic>>[];
    return items.whereType<Map>().map((item) => Map<String, dynamic>.from(item)).toList();
  }

  String _timeOf(Map<String, dynamic> event) {
    final value = DateTime.tryParse(event['timestamp']?.toString() ?? '');
    if (value == null) return 'Không rõ thời gian';
    final local = value.toLocal();
    String two(int v) => v.toString().padLeft(2, '0');
    return '${two(local.day)}/${two(local.month)}/${local.year} ${two(local.hour)}:${two(local.minute)}:${two(local.second)}';
  }

  Future<void> _acknowledge(String id) async {
    final base = widget.serverUrl.replaceAll(RegExp(r'/+$'), '');
    await http.post(Uri.parse('$base/api/events/$id/ack')).timeout(const Duration(seconds: 5));
    if (mounted) setState(() => _eventsFuture = _loadEvents());
  }

  @override
  Widget build(BuildContext context) {
    return ListView(padding: const EdgeInsets.fromLTRB(20, 18, 20, 28), children: [
      const AppHeader(title: 'Lịch sử sự kiện', subtitle: 'Cảnh báo do backend AI tạo'),
      const SizedBox(height: 22),
      Row(children: [const Expanded(child: Text('Sự kiện', style: TextStyle(fontSize: 27, fontWeight: FontWeight.w800, color: ink))), IconButton(onPressed: () => setState(() => _eventsFuture = _loadEvents()), tooltip: 'Làm mới', icon: const Icon(Icons.refresh, color: teal))]),
      const SizedBox(height: 6),
      const Text('Danh sách được lấy từ GET /api/events trên laptop.', style: TextStyle(color: mutedInk)),
      const SizedBox(height: 16),
      FutureBuilder<List<Map<String, dynamic>>>(
        future: _eventsFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) return const Padding(padding: EdgeInsets.all(24), child: Center(child: CircularProgressIndicator(color: teal)));
          if (snapshot.hasError) return Card(child: Padding(padding: const EdgeInsets.all(16), child: Text('Không tải được sự kiện. Kiểm tra kết nối backend trong phần Cài đặt.\n${snapshot.error}', style: const TextStyle(color: mutedInk, height: 1.5))));
          final events = snapshot.data ?? <Map<String, dynamic>>[];
          if (events.isEmpty) return const Card(child: Padding(padding: EdgeInsets.all(18), child: Column(children: [Icon(Icons.notifications_none, size: 38, color: mutedInk), SizedBox(height: 8), Text('Chưa có sự kiện té ngã được xác nhận.', style: TextStyle(fontWeight: FontWeight.w700, color: ink)), SizedBox(height: 5), Text('Khi bộ phát hiện sự kiện kích hoạt cảnh báo, sự kiện sẽ xuất hiện ở đây.', textAlign: TextAlign.center, style: TextStyle(color: mutedInk, height: 1.4))])));
          return Column(children: [
            for (int i = 0; i < events.length; i++) ...[
              if (i > 0) const SizedBox(height: 10),
              _LiveEventTile(event: events[i], time: _timeOf(events[i]), onAcknowledge: () async {
                final id = events[i]['id']?.toString();
                if (id == null || id.isEmpty) return;
                try { await _acknowledge(id); } catch (e) { if (context.mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Không xác nhận được sự kiện: $e'))); }
              }),
            ],
          ]);
        },
      ),
    ]);
  }
}

class _LiveEventTile extends StatelessWidget {
  final Map<String, dynamic> event;
  final String time;
  final VoidCallback onAcknowledge;
  const _LiveEventTile({required this.event, required this.time, required this.onAcknowledge});
  @override
  Widget build(BuildContext context) {
    final bool acknowledged = event['acknowledged'] == true;
    final track = event['track_id'] ?? '?';
    final confidence = ((event['confidence'] as num?)?.toDouble() ?? 0.0) * 100;
    return Card(child: Padding(padding: const EdgeInsets.all(14), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Row(children: [Icon(Icons.warning_amber_rounded, color: acknowledged ? amber : red), const SizedBox(width: 8), Expanded(child: Text('Phát hiện té ngã · ID $track', style: const TextStyle(fontWeight: FontWeight.w800, color: ink))), _pill(acknowledged ? 'ĐÃ XEM' : 'CẦN KIỂM TRA', acknowledged ? teal : red, acknowledged ? paleTeal : paleRed)]),
      const SizedBox(height: 7),
      Text('Độ tin cậy falling: ${confidence.toStringAsFixed(1)}% · $time', style: const TextStyle(color: mutedInk, fontSize: 12)),
      if (!acknowledged) ...[const SizedBox(height: 8), Align(alignment: Alignment.centerRight, child: TextButton(onPressed: onAcknowledge, child: const Text('Đánh dấu đã xem')))],
    ])));
  }
}

class SettingsPage extends StatefulWidget {
  final String serverUrl;
  final bool apiOnline;
  final String connectionMessage;
  final Map<String, dynamic>? apiStatus;
  final ValueChanged<String> onSaveServerUrl;
  final VoidCallback onCheckConnection;
  final bool checking;
  const SettingsPage({super.key, required this.serverUrl, required this.apiOnline, required this.connectionMessage, required this.apiStatus, required this.onSaveServerUrl, required this.onCheckConnection, required this.checking});
  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  late final TextEditingController _urlController;
  @override
  void initState() { super.initState(); _urlController = TextEditingController(text: widget.serverUrl); }
  @override
  void didUpdateWidget(covariant SettingsPage oldWidget) { super.didUpdateWidget(oldWidget); if (oldWidget.serverUrl != widget.serverUrl && _urlController.text != widget.serverUrl) _urlController.text = widget.serverUrl; }
  @override
  void dispose() { _urlController.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return ListView(padding: const EdgeInsets.fromLTRB(20, 18, 20, 28), children: [
      const AppHeader(title: 'Cài đặt', subtitle: 'Kết nối với máy tính chạy AI'),
      const SizedBox(height: 22),
      const Text('Backend laptop', style: TextStyle(fontSize: 25, fontWeight: FontWeight.w800, color: ink)),
      const SizedBox(height: 6),
      const Text('Điện thoại và laptop cần cùng mạng Wi-Fi ở phiên bản đầu.', style: TextStyle(color: mutedInk, height: 1.45)),
      const SizedBox(height: 16),
      Card(child: Padding(padding: const EdgeInsets.all(16), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        const Text('Địa chỉ API', style: TextStyle(fontWeight: FontWeight.w800, color: ink)),
        const SizedBox(height: 10),
        TextField(controller: _urlController, keyboardType: TextInputType.url, decoration: InputDecoration(hintText: 'http://192.168.1.10:8000', prefixIcon: const Icon(Icons.dns_outlined), filled: true, fillColor: canvas, border: OutlineInputBorder(borderRadius: BorderRadius.circular(13), borderSide: BorderSide.none), contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 14))),
        const SizedBox(height: 12),
        Row(children: [Expanded(child: OutlinedButton(onPressed: () => widget.onSaveServerUrl(_urlController.text), style: OutlinedButton.styleFrom(minimumSize: const Size.fromHeight(46), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(13))), child: const Text('Lưu địa chỉ'))), const SizedBox(width: 10), Expanded(child: FilledButton(onPressed: widget.checking ? null : () { widget.onSaveServerUrl(_urlController.text); widget.onCheckConnection(); }, style: FilledButton.styleFrom(backgroundColor: teal, minimumSize: const Size.fromHeight(46), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(13))), child: const Text('Kiểm tra API')))]),
        const SizedBox(height: 12),
        Row(crossAxisAlignment: CrossAxisAlignment.start, children: [Icon(widget.apiOnline ? Icons.check_circle : Icons.info_outline, color: widget.apiOnline ? teal : amber, size: 19), const SizedBox(width: 8), Expanded(child: Text(widget.connectionMessage, style: const TextStyle(color: mutedInk, fontSize: 13)))]),
        if (widget.apiStatus != null) ...[const SizedBox(height: 10), Container(width: double.infinity, padding: const EdgeInsets.all(10), decoration: BoxDecoration(color: paleTeal, borderRadius: BorderRadius.circular(10)), child: Text(const JsonEncoder.withIndent('  ').convert(widget.apiStatus), style: const TextStyle(fontSize: 11, color: ink)))],
      ]))),
      const SizedBox(height: 18),
      const Text('API backend đang hỗ trợ', style: TextStyle(fontSize: 19, fontWeight: FontWeight.w800, color: ink)),
      const SizedBox(height: 10),
      EndpointRow(method: 'GET', path: '/api/status', description: 'Tình trạng camera, AI và máy chủ'),
      const SizedBox(height: 8),
      EndpointRow(method: 'GET', path: '/api/events', description: 'Danh sách sự kiện té ngã'),
      const SizedBox(height: 8),
      EndpointRow(method: 'GET', path: '/stream.mjpg', description: 'Luồng video đã vẽ dự đoán'),
      const SizedBox(height: 8),
      EndpointRow(method: 'POST', path: '/api/camera/start · /stop', description: 'Điều khiển camera từ app'),
      const SizedBox(height: 8),
      EndpointRow(method: 'WS', path: '/ws', description: 'Sự kiện mới theo thời gian thực'),
      const SizedBox(height: 18),
      const Card(child: Padding(padding: EdgeInsets.all(15), child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [Icon(Icons.lock_outline, color: mutedInk), SizedBox(width: 10), Expanded(child: Text('Không đưa mật khẩu, token Firebase hoặc thông tin riêng tư vào GitHub. Khi mở truy cập ngoài mạng nội bộ, cần xác thực và mã hóa kết nối.', style: TextStyle(color: mutedInk, height: 1.5, fontSize: 13)))]))),
    ]);
  }
}

class AppHeader extends StatelessWidget {
  final String title;
  final String subtitle;
  const AppHeader({super.key, required this.title, required this.subtitle});
  @override
  Widget build(BuildContext context) => Row(children: [
    Container(width: 45, height: 45, decoration: BoxDecoration(color: ink, borderRadius: BorderRadius.circular(15)), child: const Icon(Icons.health_and_safety_outlined, color: Colors.white, size: 26)),
    const SizedBox(width: 11),
    Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Text(title, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w800, color: ink)), Text(subtitle, style: const TextStyle(fontSize: 12, color: mutedInk))])),
    Container(width: 42, height: 42, decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(14), border: Border.all(color: line)), child: IconButton(onPressed: () => ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Thông báo đẩy sẽ được tích hợp ở giai đoạn sau.'))), icon: const Icon(Icons.notifications_none, color: ink, size: 22))),
  ]);
}

class ConnectionCard extends StatelessWidget {
  final bool apiOnline;
  final String message;
  final String serverUrl;
  final bool checking;
  final VoidCallback onCheck;
  const ConnectionCard({super.key, required this.apiOnline, required this.message, required this.serverUrl, required this.checking, required this.onCheck});
  @override
  Widget build(BuildContext context) => Card(child: Padding(padding: const EdgeInsets.all(15), child: Column(children: [Row(children: [Container(width: 44, height: 44, decoration: BoxDecoration(color: apiOnline ? paleTeal : paleAmber, borderRadius: BorderRadius.circular(14)), child: Icon(apiOnline ? Icons.wifi : Icons.wifi_off, color: apiOnline ? teal : amber)), const SizedBox(width: 12), Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Text(apiOnline ? 'Backend đã kết nối' : 'Chưa kết nối laptop', style: const TextStyle(fontWeight: FontWeight.w800, color: ink)), const SizedBox(height: 3), Text(message, style: const TextStyle(fontSize: 12, color: mutedInk))])), Container(width: 9, height: 9, decoration: BoxDecoration(color: apiOnline ? teal : amber, shape: BoxShape.circle))]), const SizedBox(height: 12), Align(alignment: Alignment.centerLeft, child: Text(serverUrl, style: const TextStyle(fontSize: 11, color: mutedInk))), const SizedBox(height: 10), SizedBox(width: double.infinity, child: OutlinedButton(onPressed: checking ? null : onCheck, child: Text(checking ? 'Đang kiểm tra...' : 'Kiểm tra kết nối'), style: OutlinedButton.styleFrom(foregroundColor: teal, side: const BorderSide(color: Color(0xFFB9DCD6),), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)))))])));
}

class MetricCard extends StatelessWidget {
  final IconData icon; final String label; final String value; final String hint; final Color tint; final Color iconColor;
  const MetricCard({super.key, required this.icon, required this.label, required this.value, required this.hint, required this.tint, required this.iconColor});
  @override
  Widget build(BuildContext context) => Card(child: Padding(padding: const EdgeInsets.all(14), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Container(width: 38, height: 38, decoration: BoxDecoration(color: tint, borderRadius: BorderRadius.circular(12)), child: Icon(icon, color: iconColor)), const SizedBox(height: 12), Text(label, style: const TextStyle(color: mutedInk, fontSize: 12)), const SizedBox(height: 4), Text(value, style: const TextStyle(color: ink, fontWeight: FontWeight.w800, fontSize: 15)), const SizedBox(height: 4), Text(hint, style: const TextStyle(color: mutedInk, fontSize: 11))])));
}

class CameraPlaceholder extends StatelessWidget {
  final double height; final bool large;
  const CameraPlaceholder({super.key, required this.height, this.large = false});
  @override
  Widget build(BuildContext context) => Container(height: height, decoration: BoxDecoration(borderRadius: BorderRadius.circular(22), gradient: const LinearGradient(begin: Alignment.topLeft, end: Alignment.bottomRight, colors: [Color(0xFF1F304B), Color(0xFF101A2A)])), clipBehavior: Clip.antiAlias, child: Stack(children: [Positioned(top: -50, right: -35, child: Container(width: 180, height: 180, decoration: BoxDecoration(shape: BoxShape.circle, color: teal.withOpacity(.19)))), Positioned(bottom: -70, left: -20, child: Container(width: 190, height: 190, decoration: BoxDecoration(shape: BoxShape.circle, color: const Color(0xFF8290B7).withOpacity(.15)))), Positioned(top: 13, left: 13, child: _pill('CAM 01', Colors.white, const Color(0x26FFFFFF))), Positioned(top: 13, right: 13, child: _pill('PREVIEW DEMO', const Color(0xFFFFD790), const Color(0x2DFFCE73))), Center(child: Column(mainAxisSize: MainAxisSize.min, children: [Container(width: large ? 72 : 58, height: large ? 72 : 58, decoration: BoxDecoration(color: const Color(0xFF2B415D), borderRadius: BorderRadius.circular(22)), child: Icon(Icons.videocam_off_outlined, color: Colors.white, size: large ? 36 : 29)), const SizedBox(height: 12), const Text('Chưa có luồng video', style: TextStyle(fontWeight: FontWeight.w800, fontSize: 16, color: Colors.white)), const SizedBox(height: 5), const Text('Kết nối backend trên laptop để xem trực tiếp', textAlign: TextAlign.center, style: TextStyle(color: Color(0xFFCAD5E4), fontSize: 11))])), Positioned(bottom: 13, left: 13, right: 13, child: Row(children: [const Icon(Icons.circle, size: 8, color: Color(0xFFFFC76D)), const SizedBox(width: 6), const Expanded(child: Text('Đang chờ kết nối', style: TextStyle(color: Colors.white70, fontSize: 11))), Text('— FPS', style: TextStyle(color: Colors.white.withOpacity(.6), fontSize: 11))]))]));
}

enum EventType { alert, info, system }

class EventTile extends StatelessWidget {
  final String title; final String subtitle; final String time; final EventType type; final bool isDemo;
  const EventTile({super.key, required this.title, required this.subtitle, required this.time, required this.type, this.isDemo = false});
  @override
  Widget build(BuildContext context) {
    final Color tint = switch (type) { EventType.alert => paleRed, EventType.info => paleTeal, EventType.system => const Color(0xFFECEBFF) };
    final Color color = switch (type) { EventType.alert => red, EventType.info => teal, EventType.system => const Color(0xFF6862C8) };
    final IconData icon = switch (type) { EventType.alert => Icons.warning_amber_rounded, EventType.info => Icons.directions_walk_outlined, EventType.system => Icons.laptop_mac_outlined };
    return Card(child: Padding(padding: const EdgeInsets.all(13), child: Row(children: [Container(width: 43, height: 43, decoration: BoxDecoration(color: tint, borderRadius: BorderRadius.circular(14)), child: Icon(icon, color: color)), const SizedBox(width: 11), Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Text(title, style: const TextStyle(fontWeight: FontWeight.w800, color: ink)), const SizedBox(height: 4), Text(subtitle, style: const TextStyle(fontSize: 11, color: mutedInk))])), const SizedBox(width: 6), Column(crossAxisAlignment: CrossAxisAlignment.end, children: [Text(time, style: const TextStyle(fontSize: 10, fontWeight: FontWeight.w700, color: mutedInk)), if (isDemo) const SizedBox(height: 5), if (isDemo) _pill('MẪU', amber, paleAmber)])])));
  }
}

class SectionTitle extends StatelessWidget {
  final String title; final String? action; final VoidCallback? onAction;
  const SectionTitle({super.key, required this.title, this.action, this.onAction});
  @override
  Widget build(BuildContext context) => Row(children: [Expanded(child: Text(title, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w800, color: ink))), if (action != null) TextButton(onPressed: onAction, child: Text(action!, style: const TextStyle(color: teal, fontSize: 12, fontWeight: FontWeight.w700)))]);
}

Widget _pill(String text, Color foreground, Color background) => Container(padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 6), decoration: BoxDecoration(color: background, borderRadius: BorderRadius.circular(99)), child: Text(text, style: TextStyle(color: foreground, fontSize: 9, fontWeight: FontWeight.w800, letterSpacing: .25)));
class EndpointRow extends StatelessWidget {
  final String method;
  final String path;
  final String description;

  const EndpointRow({
    super.key,
    required this.method,
    required this.path,
    required this.description,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFFF5F7FA),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFE1E6EF)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.symmetric(
              horizontal: 8,
              vertical: 5,
            ),
            decoration: BoxDecoration(
              color: const Color(0xFFE0E7FF),
              borderRadius: BorderRadius.circular(6),
            ),
            child: Text(
              method,
              style: const TextStyle(
                fontWeight: FontWeight.w800,
                fontSize: 12,
                color: Color(0xFF3730A3),
              ),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  path,
                  style: const TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: 14,
                    color: Color(0xFF1F2937),
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  description,
                  style: const TextStyle(
                    fontSize: 12,
                    color: Color(0xFF6B7280),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}