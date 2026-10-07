import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:arborscan_app/platform_tls_trust.dart';

class _CertificateBundle extends CachingAssetBundle {
  _CertificateBundle(this.bytes);

  final List<int> bytes;
  int calls = 0;

  @override
  Future<ByteData> load(String key) async {
    calls++;
    if (key != platformRootCertificateAsset) {
      throw StateError('Unexpected asset');
    }
    return ByteData.sublistView(Uint8List.fromList(bytes));
  }
}

Future<String> _openssl() async {
  final candidates = [
    'openssl',
    if (Platform.isWindows)
      '${Platform.environment['ProgramFiles']}/Git/usr/bin/openssl.exe',
  ];
  for (final command in candidates) {
    try {
      final result = await Process.run(command, ['version']);
      if (result.exitCode == 0) return command;
    } on ProcessException {
      // Windows Git includes OpenSSL even when it is not on PATH.
    }
  }
  throw StateError('OpenSSL is required to generate synthetic TLS test keys.');
}

Future<T> _withTlsServer<T>(
  Directory directory,
  String certificate,
  Future<T> Function(int port) action,
) async {
  final serverContext = SecurityContext()
    ..useCertificateChain('${directory.path}/$certificate')
    ..usePrivateKey('${directory.path}/server.key');
  final server = await SecureServerSocket.bind(
    InternetAddress.loopbackIPv4,
    0,
    serverContext,
  );
  final connections = <SecureSocket>[];
  final subscription = server.listen(
    connections.add,
    onError:
        (Object _) {}, // Expected when the client rejects test certificates.
  );
  try {
    return await action(server.port);
  } finally {
    for (final socket in connections) {
      socket.destroy();
    }
    await subscription.cancel();
    await server.close();
  }
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  late Directory temporary;
  late List<int> testCa;
  late List<int> officialRoot;

  setUpAll(() async {
    officialRoot = await File(platformRootCertificateAsset).readAsBytes();
    temporary = await Directory.systemTemp.createTemp('arborscan-tls-test-');
    final command = await _openssl();
    Future<void> generate(List<String> arguments) async {
      final result = await Process.run(
        command,
        arguments,
        workingDirectory: temporary.path,
      );
      if (result.exitCode != 0) {
        throw StateError(
          'Synthetic TLS certificate generation failed (${arguments.first}): '
          '${result.stderr}',
        );
      }
    }

    await File('${temporary.path}/request.cnf').writeAsString(
      '[req]\ndistinguished_name=subject\nprompt=no\n'
      '[subject]\nCN=ArborScan Synthetic Test CA\n',
    );
    // Every private key is generated locally for this test and deleted below.
    await generate([
      'req',
      '-config',
      'request.cnf',
      '-x509',
      '-newkey',
      'rsa:2048',
      '-nodes',
      '-days',
      '2',
      '-subj',
      '/CN=ArborScan Synthetic Test CA',
      '-keyout',
      'ca.key',
      '-out',
      'ca.pem',
      '-addext',
      'basicConstraints=critical,CA:TRUE',
      '-addext',
      'keyUsage=critical,keyCertSign,cRLSign',
    ]);
    await generate([
      'req',
      '-config',
      'request.cnf',
      '-new',
      '-newkey',
      'rsa:2048',
      '-nodes',
      '-subj',
      '/CN=localhost',
      '-keyout',
      'server.key',
      '-out',
      'server.csr',
    ]);
    await File('${temporary.path}/server.extensions').writeAsString(
      'subjectAltName=DNS:localhost\n'
      'basicConstraints=CA:FALSE\n'
      'keyUsage=digitalSignature,keyEncipherment\n'
      'extendedKeyUsage=serverAuth\n',
    );
    for (final item in {'valid.pem': '2'}.entries) {
      await generate([
        'x509',
        '-req',
        '-in',
        'server.csr',
        '-CA',
        'ca.pem',
        '-CAkey',
        'ca.key',
        '-CAcreateserial',
        '-days',
        item.value,
        '-sha256',
        '-extfile',
        'server.extensions',
        '-out',
        item.key,
      ]);
    }
    await File('${temporary.path}/index.txt').writeAsString('');
    await File('${temporary.path}/serial.txt').writeAsString('0001\n');
    await File('${temporary.path}/ca.cnf').writeAsString(
      '[ca]\ndefault_ca=local\n[local]\nnew_certs_dir=.\n'
      'database=index.txt\nserial=serial.txt\ncertificate=ca.pem\n'
      'private_key=ca.key\ndefault_md=sha256\ndefault_days=2\n'
      'policy=subject_policy\nx509_extensions=server\n'
      '[subject_policy]\ncommonName=supplied\n'
      '[server]\nsubjectAltName=DNS:localhost\nbasicConstraints=CA:FALSE\n'
      'keyUsage=digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n',
    );
    await generate([
      'ca',
      '-batch',
      '-notext',
      '-config',
      'ca.cnf',
      '-in',
      'server.csr',
      '-out',
      'expired.pem',
      '-startdate',
      '20200101000000Z',
      '-enddate',
      '20200102000000Z',
    ]);
    testCa = await File('${temporary.path}/ca.pem').readAsBytes();
  });

  tearDownAll(() async {
    if (await temporary.exists()) await temporary.delete(recursive: true);
  });

  Future<SecurityContext> configured({bool retainTestCa = false}) async {
    final context = SecurityContext(withTrustedRoots: false);
    if (retainTestCa) context.setTrustedCertificatesBytes(testCa);
    await initializePlatformTlsTrust(
      isAndroid: true,
      bundle: _CertificateBundle(officialRoot),
      context: context,
    );
    return context;
  }

  test('the public root is packaged and can be added twice', () async {
    final context = SecurityContext(withTrustedRoots: true);
    await initializePlatformTlsTrust(isAndroid: true, context: context);
    await initializePlatformTlsTrust(isAndroid: true, context: context);
  });

  test('other platforms keep their existing trust configuration', () async {
    final bundle = _CertificateBundle([]);
    await initializePlatformTlsTrust(isAndroid: false, bundle: bundle);
    expect(bundle.calls, 0);
  });

  test('an unexpected CA cannot replace the packaged public root', () async {
    await expectLater(
      initializePlatformTlsTrust(
        isAndroid: true,
        bundle: _CertificateBundle(testCa),
        context: SecurityContext(withTrustedRoots: false),
      ),
      throwsStateError,
    );
  });

  test('adding the root preserves a previously trusted valid TLS chain',
      () async {
    final context = await configured(retainTestCa: true);
    await _withTlsServer(temporary, 'valid.pem', (port) async {
      final connection = await SecureSocket.connect(
        'localhost',
        port,
        context: context,
      );
      expect(connection.peerCertificate!.subject, contains('localhost'));
      connection.destroy();
    });
  });

  test('a trusted certificate for the wrong host remains rejected', () async {
    final context = await configured(retainTestCa: true);
    await _withTlsServer(temporary, 'valid.pem', (port) async {
      await expectLater(
        SecureSocket.connect('127.0.0.1', port, context: context),
        throwsA(isA<HandshakeException>()),
      );
    });
  });

  test('an unrelated self-signed CA remains rejected', () async {
    final context = await configured();
    await _withTlsServer(temporary, 'valid.pem', (port) async {
      await expectLater(
        SecureSocket.connect('localhost', port, context: context),
        throwsA(isA<HandshakeException>()),
      );
    });
  });

  test('an expired certificate from a trusted CA remains rejected', () async {
    final context = await configured(retainTestCa: true);
    await _withTlsServer(temporary, 'expired.pem', (port) async {
      await expectLater(
        SecureSocket.connect('localhost', port, context: context),
        throwsA(isA<HandshakeException>()),
      );
    });
  });
}
