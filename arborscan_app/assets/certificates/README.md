# Android TLS compatibility

`isrg-root-x1.pem` is the public, self-signed ISRG Root X1 certificate, downloaded
with normal HTTPS verification from
https://letsencrypt.org/certs/isrgrootx1.pem on 2026-10-07.
Its **DER SHA-256**, checked before adding it to the existing Android trust store,
is `96bcec06264976f37460779acf28c5a7cfe8a3c0aae11a8ffcee05c0bddf08c6`.
This is a public CA certificate, not a private key or a pinned server leaf.

The public API currently provides the verified default chain:
leaf → YE1 → Root YE → ISRG Root X2 (cross-signed) → ISRG Root X1.
Android 7.0 / API 24 lacks X1 and failed with `CERTIFICATE_VERIFY_FAILED`.
X1 is added before network clients are created, keeping existing platform roots,
hostname checks, certificate validity and chain verification. Modern Android
already trusts X1; adding the same public root is idempotent. Other platforms
keep their ordinary trust stores. No server or certificate-renewal changes apply.

Primary sources:

- https://letsencrypt.org/certificates/ (default YE1 chain and X1 root).
- https://letsencrypt.org/docs/certificate-compatibility/ (X1: Android ≥ 7.1.1).
- https://api.dart.dev/dart-io/SecurityContext/defaultContext.html
- https://api.dart.dev/dart-io/SecurityContext/setTrustedCertificatesBytes.html

Review the root when the API's CA chain changes and before Let's Encrypt's
published X1 trust-policy date, 2030-06-04. The certificate's `notAfter` date
(2035-06-04) is not a promise that root programs will trust it until then.
