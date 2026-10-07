import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

const platformRootCertificateAsset = 'assets/certificates/isrg-root-x1.pem';
const platformRootCertificateSha256 =
    '96bcec06264976f37460779acf28c5a7cfe8a3c0aae11a8ffcee05c0bddf08c6';

/// Adds the official public ISRG Root X1 to Android's existing trust store.
/// Android 7.0 lacks this root; the API's current public chain ends at X1.
/// Leaf, hostname, dates and chain verification continue to use Dart TLS.
Future<void> initializePlatformTlsTrust({
  bool? isAndroid,
  AssetBundle? bundle,
  SecurityContext? context,
}) async {
  if (!(isAndroid ?? (!kIsWeb && Platform.isAndroid))) return;

  final data = await (bundle ?? rootBundle).load(platformRootCertificateAsset);
  final bytes = data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
  final pem = utf8.decode(bytes);
  final certificates = RegExp(
    r'-----BEGIN CERTIFICATE-----\s*([A-Za-z0-9+/=\s]+)\s*-----END CERTIFICATE-----',
  ).allMatches(pem).toList();
  if (certificates.length != 1 ||
      pem.replaceFirst(certificates.single.group(0)!, '').trim().isNotEmpty) {
    throw StateError('The packaged public TLS root is not one certificate.');
  }
  final der = base64.decode(
    certificates.single.group(1)!.replaceAll(RegExp(r'\s'), ''),
  );
  if (sha256.convert(der).toString() != platformRootCertificateSha256) {
    throw StateError('The packaged public TLS root fingerprint differs.');
  }

  // Append to this same context: never replace the platform's trusted roots.
  (context ?? SecurityContext.defaultContext)
      .setTrustedCertificatesBytes(bytes);
}
