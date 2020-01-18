#|
 | Copyright (c) 2018-2019, University of California, Berkeley.
 | Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
 |
 | Redistribution and use in source and binary forms, with or without 
 | modification, are permitted provided that the following conditions are met:
 |
 | 1. Redistributions of source code must retain the above copyright notice, 
 | this list of conditions and the following disclaimer.
 |
 | 2. Redistributions in binary form must reproduce the above copyright notice, 
 | this list of conditions and the following disclaimer in the documentation 
 | and/or other materials provided with the distribution.
 |
 | THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 | AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 | IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 | ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 | LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 | CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 | SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 | INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 | CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 | ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 | POSSIBILITY OF SUCH DAMAGE.
 |#

#lang rosette

(require "util.rkt" "cuda.rkt" "cuda-synth.rkt")

(define struct-size 3)
(define n-block 1)

(define (create-IO warpSize)
  (set-warpSize warpSize)
  (define block-size (* 1 warpSize))
  (define array-size (* n-block block-size))
  (define I-sizes (x-y-z (* array-size struct-size)))
  (define I (create-matrix I-sizes gen-uid))
  (define O (create-matrix I-sizes))
  (define O* (create-matrix I-sizes))
  (values block-size I-sizes I O O*))

(define (run-with-warp-size spec kernel w)
  (define-values (block-size I-sizes I O O*)
  (create-IO w))

  (define c (gcd struct-size warpSize))
  (define a (/ struct-size c))
  (define b (/ warpSize c))

  (run-kernel spec (x-y-z block-size) (x-y-z n-block) I O a b c)
  (run-kernel kernel (x-y-z block-size) (x-y-z n-block) I O* a b c)
  (define ret (equal? O O*))
  ;(pretty-display `(O ,O))
  ;(pretty-display `(O* ,O*))
  ret)

(define (AOS-load-spec threadId blockID blockDim I O a b c)
  (define I-cached (create-matrix-local (x-y-z struct-size)))
  (define warpID (get-warpId threadId))
  (define offset (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (define gid (get-global-threadId threadId blockID))
  (global-to-local I I-cached
                 (x-y-z struct-size)
                 offset (x-y-z (* warpSize struct-size)) #f)
  (local-to-global I-cached O
                      (x-y-z 1) offset (x-y-z (* warpSize struct-size)) #f #:round struct-size)
  )

(define (print-vec x)
  (format "#(~a)" (string-join (for/list ([xi x]) (format "~a" xi)))))

;; Sketch that uses column-row-column shuffles.
(define (AOS-load-sketch threadId blockID blockDim I O a b c)
  (define I-cached (create-matrix-local (x-y-z struct-size)))
  (define O-cached (create-matrix-local (x-y-z struct-size)))
  
  (define localId (modulo (get-x threadId) 32))
  (define offset (* struct-size (- (+ (* blockID blockDim) (get-x threadId)) localId)))
  
  (global-to-local I I-cached
                 (x-y-z 1)
                 offset
                 (x-y-z (* warpSize struct-size)) #f #:round struct-size)

  ;; column shuffle
  (define I-cached2 (permute-vector I-cached struct-size
                                    (lambda (i) (?sw-xform i struct-size localId warpSize))))

  (define V I-cached2)
  (assert (equal? 65 (vector-ref (vector-ref V 0) 1)))
  (assert (equal? 1  (vector-ref (vector-ref V 0) 2)))

  (assert (equal? 3 (vector-ref (vector-ref V 2) 0)))
  (assert (equal? 4 (vector-ref (vector-ref V 3) 2)))

  (assert (equal? 38 (vector-ref (vector-ref V 5) 1)))
  (assert (equal? 10 (vector-ref (vector-ref V 9) 2)))

  (assert (equal? 12 (vector-ref (vector-ref V 11) 0)))
  (assert (equal? 50 (vector-ref (vector-ref V 17) 1)))

  (assert (equal? 62 (vector-ref (vector-ref V 29) 1)))
  (assert (equal? 64 (vector-ref (vector-ref V 31) 2)))

  (assert (equal? 78 (vector-ref (vector-ref V 13) 0)))
  (assert (equal? 28 (vector-ref (vector-ref V 27) 2)))

  (assert (equal? 9 (vector-ref (vector-ref V 8) 0)))
  (assert (equal? 17 (vector-ref (vector-ref V 16) 1)))

  (assert (equal? 81 (vector-ref (vector-ref V 16) 0)))
  (assert (equal? 89 (vector-ref (vector-ref V 24) 1)))

  (assert (equal? 65 (vector-ref (vector-ref V 0) 1)))
  (assert (equal? 35 (vector-ref (vector-ref V 2) 1)))

  (assert (equal? 33 (vector-ref (vector-ref V 0) 0)))
  (assert (equal? 66 (vector-ref (vector-ref V 1) 0)))

  (assert (equal? 73 (vector-ref (vector-ref V 8) 2)))
  (assert (equal? 49 (vector-ref (vector-ref V 16) 2)))

  ;; row shuffle
  (for ([i struct-size])
    (let* ([lane (?sw-xform localId warpSize i struct-size)]
           [x (shfl (get I-cached2 (@dup i)) lane)]
           )
      (set O-cached (@dup i) x))
    )
  
  (define V1 O-cached)
  (assert (equal? 1 (vector-ref (vector-ref V1 0) 2)))
  (assert (equal? 4 (vector-ref (vector-ref V1 1) 2)))

  (assert (equal? 26 (vector-ref (vector-ref V1 8) 1)))
  (assert (equal? 25 (vector-ref (vector-ref V1 8) 2)))

  (assert (equal? 66 (vector-ref (vector-ref V1 21) 0)))
  (assert (equal? 72 (vector-ref (vector-ref V1 23) 0)))

  (assert (equal? 95 (vector-ref (vector-ref V1 31) 1)))
  (assert (equal? 94 (vector-ref (vector-ref V1 31) 2)))

  (assert (equal? (??) (vector-ref (vector-ref V1 17) 0)))
  (assert (equal? (??) (vector-ref (vector-ref V1 9) 2)))
  
  ;; column shuffle
  (define O-cached2 (permute-vector O-cached struct-size
                                    (lambda (i) (?sw-xform i struct-size localId warpSize))))
  

  (local-to-global O-cached2 O
                      (x-y-z 1)
                      offset
                      (x-y-z (* warpSize struct-size)) #f #:round struct-size)
  )

;; Sketch that uses row-column-row shuffles.
(define (AOS-load-rcr-sketch threadId blockID blockDim I O a b c)
  (define I-cached (create-matrix-local (x-y-z struct-size)))
  
  (define localId (modulo (get-x threadId) 32))
  (define offset (* struct-size (- (+ (* blockID blockDim) (get-x threadId)) localId)))

  ;; load with (row) shuffle
  (global-to-local
   I
   I-cached
   (x-y-z 1)
   offset
   (x-y-z (* warpSize struct-size))
   #f #:round struct-size
   #:shfl (lambda (localId i) (?sw-xform localId warpSize i struct-size)))

  ;; column shuffle
  (define O-cached (permute-vector I-cached struct-size
                                   (lambda (i) (?sw-xform i struct-size localId warpSize))))

  ;; store with (row) shuffle
  (local-to-global
   O-cached
   O
   (x-y-z 1)
   offset
   (x-y-z (* warpSize struct-size))
   #f #:round struct-size
   #:shfl (lambda (localId i)
            (?sw-xform localId warpSize i struct-size)))
  )

(define (test)
  (for ([w (list 32)])
    (let ([ret (run-with-warp-size AOS-load-spec AOS-load-sketch w)])
      (pretty-display `(test ,w ,ret))))
  )
;(test)

(define (synthesis)
  (pretty-display "solving...")
  (assert (andmap (lambda (w) (run-with-warp-size AOS-load-spec AOS-load-sketch w))
                                           (list 32)))
  (define cost (get-cost))
  
  (define sol (time (optimize #:minimize (list cost) #:guarantee (assert #t))))

  (define this-cost (evaluate cost sol))
  (print-forms sol)
  (pretty-display `(cost ,this-cost))
  )
(define t0 (current-seconds))
(synthesis)
(define t1 (current-seconds))
(- t1 t0)
